function [ im, msg ] = read_image_file( imgfile, imgbytes )

im = [];
msg = false;
if imgbytes < 1000
    msg = false;
else
    try
        im = imread(imgfile);
        [~, ~, nchannels] = size(im);
        if nchannels ~=3
            msg = false;
        elseif isa(im, 'uint16')
            im = uint8(double(im) / 65535 * 255);
            msg = true;
        else
            msg = true;
        end
    catch
        msg = false;
    end
end

end

