clc
clear
close all

%%
addpath('../');
%% load morphable models
vertex_data = load('../../3dmm/vertex_code.mat');
shape_model = load('../../3dmm/Model_Shape.mat');
expression_model = load('../../3dmm/Model_Expression.mat');
vertex_code = vertex_data.vertex_code;
mu_shape = shape_model.mu_shape;
w = shape_model.w;
tri = shape_model.tri;
% sigma = shape_model.sigma;
mu_exp = expression_model.mu_exp;
w_exp = expression_model.w_exp;
% sigma_exp = expression_model.sigma_exp;
num_pose_param = 7;
num_shape_param = size(w, 2);
num_exp_param = size(w_exp, 2);
mu = mu_shape + mu_exp;
im_size = 200;
beta = 1.0;
%%

im_face = imread('../../data/vggface/face_images/Jim_Sturgess/00000015.jpg');
im_face = imresize(im_face, [im_size, im_size]);
% 
[ pose_param, shape_param, exp_param ] = get_random_params( im_size, num_shape_param, num_exp_param, beta);
[pncc, masked_img] = Project2D(mu, w, w_exp, tri, vertex_code, im_face, pose_param, shape_param, exp_param);
imshow(pncc);
figure;
imshow(masked_img)

addpath('../');
