%     demo for CNN-TD Detection algorithm
%--------------Brief description-------------------------------------------
%
% 
% This demo implements the  CNN-TD hyperspectral image Detection [1]
%
%
% More details in:
%
% [1] W. Li, G. Wu, Q. Du. Transferred Deep Learning for Anomaly Detection in % Hyperspectral Imagery
% IEEE Geoscience and Remote Sensing Letters
% 14(5), 597-601, 2017
%
% contact: liwei089@ieee.org (Wei Li)
% contact: 495482236@qq.com (Guodong Wu)

clear all;  clc; 
close all

load z_moffett
z = reshape(z', 512, 512, 224);
DataTest_ori = z./max(z(:));
[rows cols, bands] = size(z);
load moffett_mask05
M = rows * cols;
[a b] = size(mask);
X = reshape(DataTest_ori, rows*cols, bands);
X = X';  % num_dim x num_sam
[N M] = size(X);      % num_dim x num_sam
   
% show the original image 
A1 = DataTest_ori;
X = zeros(rows, cols);
X(:, :, 1) = histeq(A1(:, :, 60));
X(:, :, 2) = histeq(A1(:, :, 30));
X(:, :, 3) = histeq(A1(:, :, 2));
figure, imagesc(X); axis image; colorbar; impixelinfo
A = reshape(mask, rows, cols);
figure, imagesc(A); axis image; colorbar; impixelinfo
