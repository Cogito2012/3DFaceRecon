function [ ] = close_matlabpool

if isempty(gcp('nocreate'))==0
    delete(gcp('nocreate'));
end

end
