function simpleSave( fn, varargin )
%SIMPLESAVE save variables in the simpleSave format
%   simpleSave( filename, vars... )
%
% INPUT:
%   filename -- string -- file path for the file to write, without the
%       suffixes .sss, .ssd
%   vars -- names of variables to save
%
% The variables are saves as fields of the a top-level simpleSave REC
% Data is stored in two files -- .sss and .ssd containing structure and
% binary data. 
%
% The following option flags are supported:
%   '-CTX' -- emit context debug comments in the .ssd file. 
%
% SimpleSave data can contain arrays of arbitrary numeric types,
%   struct arrays, cell arrays and char arrays. Function handles and inline
%   functions are converted to a string representation.
% $Revision: 1.2 $
% By Shai Revzen, Berkeley 2008
    global identity_foo
    identity_foo = 1;
    
    ctx.plan=cell(0,2);
    ctx.msg = '%% CTX(';
    ctx.stack = length(ctx.msg);
    ctx.ctxMsg = false;
    arg={};
    % Collect arguments while parsing any options
    for k=1:length(varargin)
        if varargin{k}(1)=='-'
            switch varargin{k}(2:end);
                case 'CTX';
                    ctx.ctxMsg = true;
                otherwise
                    error('simpleSave:unknownOption',['Unknown option "',varargin{k},'"']);
            end
        else
            arg{end+1} = varargin{k};
        end
    end
	if isempty(arg)
        error('simpleSave:empty','must give list of things to save');
	end	
    data = struct;
	for k=1:length(arg)
        data = setfield( data, arg{k}, evalin('caller',arg{k}) );
	end
	try;
        ctx.f_dcr=fopen([fn '.sss'],'w');
        ctx.f_raw=fopen([fn '.ssd'],'w');
	catch IOE;
        IOE
        fclose(ctx.f_dcr);
        fclose(ctx.f_raw);
        return;
	end
    ctx = simpleSaveStruct( ctx, data );
    try;
        emitData(ctx);
    catch IOE;
        IOE
        warning('simpleSave:corrupt','Error writing -- files may be corrupt');
    end
    fclose(ctx.f_dcr);
    fclose(ctx.f_raw);
return

function ctx = emitData( ctx )
  fprintf(ctx.f_dcr,'DECL_ENDS(1);\n\n');  
  for k=1:size(ctx.plan,1)
      fwrite(ctx.f_raw,ctx.plan{k,1},ctx.plan{k,2});
  end
return

function ctx = planEmit( ctx, dat, fmt )
  ctx.plan{end+1,1} = dat;
  ctx.plan{end,2} = fmt;
return

function ctx = pushCtx( ctx, nm )
  ctx.stack(end+1) = length(ctx.msg);
  ctx.msg = [ctx.msg, nm,','];
return

function ctx = popCtx( ctx )
  ctx.msg = ctx.msg(1:ctx.stack(end));
  ctx.stack = ctx.stack(1:end-1);
return

function ctx = showCtx( ctx, fmt, varargin )
  if ctx.ctxMsg>0
      fprintf(ctx.f_dcr, [ctx.msg(1:end-1),'); ',fmt], varargin{:} );
  end
return

function ctx = emitShape( ctx, sz )
  fprintf(ctx.f_dcr,'SZ(%d',sz(1));
  fprintf(ctx.f_dcr,',%d',sz(2:end));
  fprintf(ctx.f_dcr,')');
return

function ctx = emitName( ctx, nm )
  if isempty(nm)
      fprintf(ctx.f_dcr,'NAME('''')');
  else
      fprintf(ctx.f_dcr,'NAME(''%s'')',nm);
  end
return

function ctx = simpleSaveAny( ctx, dat )
  if ischar(dat)
      ctx = simpleSaveChar(ctx,dat);
  elseif isnumeric(dat)
      ctx = simpleSaveArray(ctx,dat);
  elseif isstruct(dat) 
      ctx = simpleSaveStruct(ctx,dat);
  elseif iscell(dat)
      ctx = simpleSaveCell(ctx,dat);
  elseif islogical(dat)
      ctx = simpleSaveBool(ctx,dat);
  else switch class(dat);
      case 'function_handle';
          ctx = simpleSaveFunction(ctx,dat);
      case 'inline';
          ctx = simpleSaveInline(ctx,dat);          
      otherwise;
          error('simpleSave:unknown',sprintf('Cannot save class "%s"',class(dat)));
      end
  end
return

function nm = newUniqueID
  global identity_foo
  nm = num2str(identity_foo);
  identity_foo = identity_foo + 1;
  return
  
function ctx = simpleSaveFunction( ctx, dat )
  fprintf(ctx.f_dcr,'FUNCTION_HANDLE(''%s'');\n',func2str(dat));
  return;

function ctx = simpleSaveInline( ctx, dat )
  s = cell2struct( struct2cell(dat), fieldnames(dat) );
  args = [s.args ,repmat(',',size(s.args,1),1)].';
  args = strrep(args(1:end-1),' ','');
  expr = strrep(s.expr,'\','\\');
  expr = strrep(expr,'''','\x27');
  fprintf(ctx.f_dcr,'INLINE(''%s'',''%s'');\n',args,expr);
  return;
  
function ctx = simpleSaveChar( ctx, dat ) 
  fprintf(ctx.f_dcr,'CHAR(');
  emitShape(ctx,size(dat));
  ctx = planEmit(ctx,dat,'char');
  fprintf(ctx.f_dcr,');\n');
  return;

function ctx = simpleSaveBool( ctx, dat ) 
  fprintf(ctx.f_dcr,'BOOL(');
  emitShape(ctx,size(dat));
  ctx = planEmit(ctx,dat,'uint8');
  fprintf(ctx.f_dcr,');\n');
  return;
return

function ctx = simpleSaveArray( ctx, dat )
  if any(imag(dat(:)))
      fprintf(ctx.f_dcr,'COMPLEX_%s(',class(dat));
      emitShape(ctx,size(dat));
      ctx = planEmit(ctx,real(dat),class(dat));
      ctx = planEmit(ctx,imag(dat),class(dat));
      fprintf(ctx.f_dcr,');\n');
  else
      fprintf(ctx.f_dcr,'REAL_%s(',class(dat));
      emitShape(ctx,size(dat));
      ctx = planEmit(ctx,dat,class(dat));
      fprintf(ctx.f_dcr,');\n');
  end
return        
          
function ctx = simpleSaveStruct( ctx, dat )
    nm = newUniqueID;
    ctx = pushCtx( ctx, nm );
	fprintf(ctx.f_dcr,'REC(');
	emitShape(ctx,size(dat));
	fn = fieldnames(dat);
	if ~isempty(fn)
        fprintf(ctx.f_dcr,',');
		emitName(ctx,fn{1});
		for k=2:length(fn)
          fprintf(ctx.f_dcr,',');
          emitName(ctx, fn{k});
		end
        fprintf(ctx.f_dcr,'); %% BEGIN_REC_ID(%s)\n',nm);
		dat=dat(:);
        if ~isempty(dat)
            for di=1:length(dat)
                ctx = showCtx(ctx,' AT(%d);\n',di);
                for k=1:length(fn)
                    ctx = showCtx(ctx,' FIELD(''%s'')\n',fn{k});
                    ctx = simpleSaveAny( ctx, getfield(dat(di),fn{k}) );
                end
            end
        end
    else
        fprintf(ctx.f_dcr,'); %%COMMENT(''struct without fields'')\n');
    end
    ctx = showCtx(ctx,'END_REC_ID(%s)\n',nm);
    ctx = popCtx(ctx);   
return

function ctx = simpleSaveCell( ctx, dat )
    nm = newUniqueID;
    ctx = pushCtx( ctx, nm );
	fprintf(ctx.f_dcr,'OBJ(');
	emitShape(ctx,size(dat));
    fprintf(ctx.f_dcr,'); %% BEGIN_OBJ_ID(%s)\n', nm);
	dat=dat(:);
    if ~isempty(dat)
        for di=1:length(dat)
            ctx = showCtx(ctx,' AT(%d);\n',di);
            ctx = simpleSaveAny( ctx, dat{di} );
        end
    end
    ctx = showCtx(ctx,' END_OBJ_ID(%s)\n',nm);
    ctx = popCtx(ctx);   
return
