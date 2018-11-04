function [code_map, masked_img] = Project2D( mu, w, w_exp, tri, vertex_code, img, Pose_Para, Shape_Para, Exp_Para )
[phi, gamma, theta, t3d, f] = ParaMap_Pose(Pose_Para);
R = RotationMatrix(phi, gamma, theta);
P = [1 0 0; 0 1 0];

alpha = Shape_Para;
alpha_exp = Exp_Para;
express = w_exp * alpha_exp; express = reshape(express, 3, length(express)/3);
shape = mu + w * alpha; shape = reshape(shape, 3, length(shape)/3);
vertex = shape + express;

ProjectVertex = f * R * vertex + repmat(t3d, 1, size(vertex, 2));

%DrawSolidHead(ProjectVertex, tri);
[height, width, nChannels] = size(img);
code_map = zeros(height, width, nChannels);
masked_img = zeros(height, width, nChannels);
[code_map, tri_ind] = Mex_ZBuffer(double(ProjectVertex), double(tri), double(vertex_code), code_map);
code_map = InvertImage(code_map);

mask = flipud(tri_ind);
mask(find(mask > 0)) = 1;
masked_img = uint8(repmat(mask, [1, 1, nChannels])) .* img;
end

