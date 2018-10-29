function [ box_valid ] = check_box( box, height, width )
%  box: [left top right bottom]
box_valid = [floor(box(1)), floor(box(2)), floor(box(3)), floor(box(4))];

if box_valid(1) < 1, box_valid(1) = 1; end
if box_valid(1) > width, box_valid(1) = width; end
if box_valid(2) < 1, box_valid(2) = 1; end
if box_valid(2) > height, box_valid(2) = height; end

if box_valid(3) < 1, box_valid(3) = 1; end
if box_valid(3) > width, box_valid(3) = width; end
if box_valid(4) < 1, box_valid(4) = 1; end
if box_valid(4) > height, box_valid(4) = height; end

end

