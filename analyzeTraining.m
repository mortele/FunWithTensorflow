%% Clean up
close all;
clear all;
clc;
format;


%% Load in training data
fileName    = 'TrainingData/14.09-23.49.40/meta.dat';
inFile      = fopen(fileName);
data        = textscan(inFile, '%f%f', 'HeaderLines',1,...
                       'Delimiter', '\n', 'CollectOutput',1);
fclose(inFile);
data        = cell2mat(data);


%% Plot training error
trainingCost    = data(:,1);
testCost        = data(:,2);
semilogy(trainingCost(1:10:end));
xlabel('epoch / 10', 'FontSize', 16);
ylabel('Test cost / number of points', 'FontSize', 16);
