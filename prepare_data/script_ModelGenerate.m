load('../3dmm/01_MorphableModel.mat');
load('../3dmm/model_info.mat');
trimIndex1 = [3*trimIndex-2, 3*trimIndex-1, 3*trimIndex]';
trimIndex1 = trimIndex1(:);

mu_shape = shapeMU(trimIndex1);
w = shapePC(trimIndex1, :);

tex = texMU(trimIndex1);
tex = reshape(tex, 3, length(tex)/3);
w_tex = texPC(trimIndex1, :);
alpha_tex = texEV;

segbin = segbin(trimIndex, :);

save('../3dmm/Model_Shape.mat', 'mu_shape', 'w', 'tex', 'tri', 'w_tex', 'alpha_tex');


