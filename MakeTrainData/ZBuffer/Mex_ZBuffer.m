function [img, tri_ind] = Mex_ZBuffer(projectedVertex, tri, texture, img_src)
    [height, width, nChannels] = size(img_src);
    nver = size(projectedVertex,2);
    ntri = size(tri,2);
    [img, tri_ind] = Mex_ZBufferC(double(projectedVertex), double(tri), double(texture), img_src, nver, ntri, width, height, nChannels);
end
%% Matlab Ver.
% function [img] = ZBuffer(s2d, vertex, tri, texture, img_src)
% [height, width, nChannels] = size(img_src);
% img = img_src;
% s2d(2,:) = height - s2d(2,:) + 1;
% % for every triangles, find the point it covers
% imgr = zeros(height, width , 1);
% n = size(tri, 2);
% 
% 
% point1 = s2d(:, tri(1, :)); % A point
% point2 = s2d(:, tri(2, :)); % B point
% point3 = s2d(:, tri(3, :)); % C point
% 
% cent3d = (vertex(:, tri(1, :)) + vertex(:, tri(2, :)) + vertex(:, tri(3, :))) / 3;
% r = cent3d(1,:).^2 + cent3d(2,:).^2 + cent3d(3,:).^2;
% 
% tritex = (texture(:, tri(1, :)) + texture(:, tri(2, :)) + texture(:, tri(3, :))) / 3;
% 
% for i = 1:n
%     i
%     pt1 = point1(:,i);
%     pt2 = point2(:,i);
%     pt3 = point3(:,i);
%     
%     umin = ceil(min([pt1(1); pt2(1); pt3(1)]));
%     umax = floor(max([pt1(1), pt2(1), pt3(1)]));
%     
%     vmin = ceil(min([pt1(2), pt2(2), pt3(2)]));
%     vmax = floor(max([pt1(2), pt2(2), pt3(2)]));
%     
%     if(umax < umin || vmax < vmin || umax > width || umin < 1 || vmax > height || vmin < 1)
%         continue;
%     end
%     
%     
%     for u = umin : umax
%         for v = vmin : vmax
%             if( imgr(v,u) < r(i) && triCpoint([u;v], pt1, pt2, pt3))
%                 imgr(v,u) = r(i);
%                 img(v,u,:) = tritex(:,i);
%             end
%         end
%     end
% end
% 
% imshow(img/255);
%     
% end
% 
% % judge if this triangle cover the point
% function [state] = triCpoint(point, pt1, pt2, pt3)
% 
% state = 1;
% 
% v0 = pt3 - pt1; %C - A
% v1 = pt2 - pt1; %B - A 
% v2 = point - pt1;
% 
% dot00 = v0' * v0;
% dot01 = v0' * v1;
% dot02 = v0' * v2;
% dot11 = v1' * v1;
% dot12 = v1' * v2;
% 
% inverDeno = 1 / (dot00 * dot11 - dot01 * dot01);
% 
% u = (dot11 * dot02 - dot01 * dot12) * inverDeno;
% 
% if(u < 0 || u > 1)
%     state = 0;
%     return;
% end
% 
% v = (dot00 * dot12 - dot01 * dot02) * inverDeno;
% if(v < 0 || v > 1)
%     state = 0;
%     return;
% end
% 
% state = u + v <= 1;
% 
% end
% 


% function [img] = Z_Buffer(s2d, vertex, tri, texture, img_src)
% aaa = 1;
% [height, width, nChannels] = size(img_src);
% img = img_src;
% s2d(2,:) = height - s2d(2,:) + 1;
% % for every triangles, find the point it covers
% imgr = zeros(height, width , 1);
% n = size(tri, 2);
% 
% 
% point1 = s2d(:, tri(1, :)); % A point
% point2 = s2d(:, tri(2, :)); % B point
% point3 = s2d(:, tri(3, :)); % C point
% 
% cent3d = (vertex(:, tri(1, :)) + vertex(:, tri(2, :)) + vertex(:, tri(3, :))) / 3;
% r = cent3d(1,:).^2 + cent3d(2,:).^2 + cent3d(3,:).^2;
% 
% tritex = (texture(:, tri(1, :)) + texture(:, tri(2, :)) + texture(:, tri(3, :))) / 3;
% 
% for i = 1:n
%     i
%     pt1 = point1(:,i);
%     pt2 = point2(:,i);
%     pt3 = point3(:,i);
%     
%     umin = ceil(min([pt1(1); pt2(1); pt3(1)]));
%     umax = floor(max([pt1(1), pt2(1), pt3(1)]));
%     
%     vmin = ceil(min([pt1(2), pt2(2), pt3(2)]));
%     vmax = floor(max([pt1(2), pt2(2), pt3(2)]));
%     
%     if(umax < umin || vmax < vmin || umax > width || umin < 1 || vmax > height || vmin < 1)
%         continue;
%     end
%     
%     
%     for u = umin : umax
%         for v = vmin : vmax
%             if( imgr(v,u) < r(i) && triCpoint([u;v], pt1, pt2, pt3))
%                 imgr(v,u) = r(i);
%                 img(v,u,:) = tritex(:,i);
%             end
%         end
%     end
% end
% 
% imshow(img/255);
%     
% end
% 
% % judge if this triangle cover the point
% function [state] = triCpoint(point, pt1, pt2, pt3)
% 
% state = 1;
% 
% v0 = pt3 - pt1; %C - A
% v1 = pt2 - pt1; %B - A 
% v2 = point - pt1;
% 
% dot00 = v0' * v0;
% dot01 = v0' * v1;
% dot02 = v0' * v2;
% dot11 = v1' * v1;
% dot12 = v1' * v2;
% 
% inverDeno = 1 / (dot00 * dot11 - dot01 * dot01);
% 
% u = (dot11 * dot02 - dot01 * dot12) * inverDeno;
% 
% if(u < 0 || u > 1)
%     state = 0;
%     return;
% end
% 
% v = (dot00 * dot12 - dot01 * dot02) * inverDeno;
% if(v < 0 || v > 1)
%     state = 0;
%     return;
% end
% 
% state = u + v <= 1;
% 
% end
% 
% 



