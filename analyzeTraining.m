%% Clean up
close all;
clear all;
clc;
format;


%% Load in training data
%fileName    = 'TrainingData/16.09-08.30.52/meta.dat';
fileName    = 'TrainingData/17.09-02.05.34/meta.dat';
%fileName    = 'meta1.dat';
inFile      = fopen(fileName);
data        = textscan(inFile,  '%f%f', 'HeaderLines', 1, ...
                                'Delimiter', '\n', 'CollectOutput', 1);
fclose(inFile);
data = cell2mat(data);


%% Plot training error
skip            = 10;
trainingCost    = data(:,1);
testCost        = data(:,2);
epoch           = linspace(0,size(data, 1), size(data,1));
trainingSetSize = 1e7;
semilogy(epoch(1:skip:end).*trainingSetSize, ...
         trainingCost(1:skip:end));
xlabel('Total training data points used', 'FontSize', 16, ...
       'interpreter', 'latex');
ylabel('$\|$NN$(r)-$LJ$(r)\|$ $/$ number of points ($L^2$ norm)', 'FontSize', 16, ...
       'interpreter', 'latex');
