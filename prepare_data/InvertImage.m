function [des] = InvertImage(src)

des = src;
[temp, temp, nChannels] = size(src);
for i = 1:nChannels
    des(:,:,i) = flipud(src(:,:,i));
end

end

