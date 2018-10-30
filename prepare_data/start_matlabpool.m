function [ pool ] = start_matlabpool( size )

pool=[];
isstart = 0;
if isempty(gcp('nocreate'))==1
    isstart = 1;
end
if isstart==1
    if nargin==0
        pool=parpool('local');
    else
        try
            pool=parpool('local',size);%matlabpool('open','local',size);
        catch ce
            pool=parpool('local');%matlabpool('open','local');
            size = pool.NumWorkers;
            display(ce.message);
            display(strcat('restart. wrong  size=',num2str(size)));
        end
    end
else
    disp('matlabpool has started');
    if nargin==1
        close_matlabpool;
        start_matlabpool(size);
    else
        start_matlabpool();
    end
end


end

