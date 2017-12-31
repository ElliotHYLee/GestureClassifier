clc, close all, clear;

LT_Test = importdata('LTTest.mat');
LT_Train = importdata('LTTrain.mat');

idle_test = importdata('idleTest.mat');
idle_train = importdata('idleTrain.mat');

retreatTest = importdata('retreatTest.mat');
retreatTrain = importdata('retreatTrain.mat');

proceedTest = importdata('proceedTest.mat');
proceedTrain = importdata('proceedTrain.mat');

seqNum  = 30;
lastCol = seqNum*6 +1 ;

sq_idle_train = getSequencial(idle_train, seqNum);
sq_idle_train(:,lastCol) = 1;
sq_idle_test = getSequencial(idle_test, seqNum);
sq_idle_test(:,lastCol) = 1;

sq_LT_train = getSequencial(LT_Train, seqNum);
sq_LT_train(:,lastCol) = 2;
sq_LT_test = getSequencial(LT_Test, seqNum);
sq_LT_test(:,lastCol) = 2;

sq_proceed_train = getSequencial(proceedTrain, seqNum);
sq_proceed_train(:,lastCol) = 3;
sq_proceed_test = getSequencial(proceedTest, seqNum);
sq_proceed_test(:,lastCol) = 3;

sq_moveBack_train = getSequencial(retreatTrain, seqNum);
sq_moveBack_train(:,lastCol) = 4;
sq_moveBack_test = getSequencial(retreatTest, seqNum);
sq_moveBack_test(:,lastCol) = 4;


trainSet = [
            sq_idle_train;
            sq_LT_train;
            sq_proceed_train;
            sq_moveBack_train];

testSet = [
            sq_idle_test;
            sq_LT_test;
            sq_proceed_test;
            sq_moveBack_test];






