function [phi, gamma, theta, t3d, f] = ParaMap_Pose(para_Pose)
phi = para_Pose(1);
gamma = para_Pose(2);
theta = para_Pose(3);
t3dx = para_Pose(4);
t3dy = para_Pose(5);
t3dz = para_Pose(6);
f = para_Pose(7);

t3d = [t3dx; t3dy; t3dz];
end

