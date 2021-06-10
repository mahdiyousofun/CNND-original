clear all; clc;
close all;

load z_moffett.mat
z = z';
n = size(z, 2);
% z = (z - min(z(:))) / max(z(:));
a = z(randi(size(z, 1)), :);
plot(1:n, a);
hold on

load image.mat
z1 = pixels';
n = size(z1, 2);
b = z1(randi(size(z1, 1)), :);
plot(1:n, b);