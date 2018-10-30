function [ pose_param, shape_param, exp_param ] = get_random_params( im_size, num_shape_param, num_exp_param, beta)

phi = (-75 + 120 * rand(1)) * pi/180;  % [-75, 45]
gamma = (-90 + 180 * rand(1)) * pi/180;  % [-90, 90]
theta = (-30 + 60 * rand(1)) * pi/180;  % [-20, 20]
focal_factor = rand(1,1) * 1e-3;
t3d = [rand(2, 1) * (60); 0];
pose_param = [phi; gamma; theta; t3d; focal_factor];
pose_param = beta * [0; 0; 0; im_size/2; im_size/2; 0; 0.001] + (1-beta)*pose_param;
shape_param = rand(num_shape_param, 1) * 1e04;
exp_param = -1.5 + 3 * rand(num_exp_param, 1);

% phi = (-10 + 20 * normrnd(0, 0.1)) * pi/180;  % [-60, 45]
% gamma = (-20 + 40 * normrnd(0, 0.1)) * pi/180;  % [-90, 90]
% theta = (-5 + 10 * normrnd(0, 0.1)) * pi/180;  % [-20, 20]
% focal_factor = normrnd(0, 1) * 1e-3;
% t3d = normrnd(0, 1, 3, 1) * (im_size / 2);
% pose_param = [phi; gamma; theta; t3d; focal_factor];
% shape_param = normrnd(0, 1, num_shape_param, 1) * 1e04;
% exp_param = -1.5 + 3 * normrnd(0, 1, num_exp_param, 1);


end

