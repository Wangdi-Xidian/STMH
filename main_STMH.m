function [Bi_Ir,Bt_Tr,Bi_Ie,Bt_Te,traintime,testtime] = main_STMH(I_tr, T_tr, I_te, T_te, bits, lambda, mu, gamma, numzeros,maxIter)
% Notation:
% I_tr: data matrix of image, each row is a sample vector
% T_tr: data matrix of text, each row is a sample vector
% lambda: trade off between different views
% mu: trade off between construction and linear projection
% gamma: parameter to control the model complexity
% numzeros: number of zeros in each hash code

% Reference:
% Di Wang, Xinbo Gao, Xiumei Wang, Lihuo He
% "Semantic topic multimodal hashing for cross-media retrieval"
% International Joint Conference on Artificial Intelligence (IJCAI),
% pp.3890¨C3896, 2015.
% wangdi.wandy@gmail.com
%% Parameter Setting
if ~exist('lambda','var')
    lambda = 0.5;
end
if ~exist('mu','var')
    mu = 0.000001;
end
if ~exist('gamma','var')
    gamma = 0.000001;
end
if ~exist('numzeros','var')
    numzeros = round(bits/2);
end
if ~exist('maxIter','var')
    maxIter = 5;
end
inXCell = cell(2,1);
inXCell{1,1} = I_tr';
inXCell{2,1} = T_tr';

inPara.maxIter = maxIter;
inPara.thresh = 0.01;
inPara.bits = bits;
inPara.r = [lambda,1-lambda];
inPara.mu = mu;
inPara.gamma = gamma;

TrTime1 = cputime;
[G,outFCell,R] = multi_STMH(inXCell, inPara, numzeros);
Bt_Tr = compactbit(G{2});
Bi_Ir = Bt_Tr;

TrTime2 = cputime;
traintime = TrTime2-TrTime1;

TeIime1 = cputime;
Y_Te  = STMH_coding(T_te', outFCell{2},numzeros);
Bt_Te = compactbit(Y_Te);
Y_Ie = I_te*outFCell{1}/(outFCell{1}'*outFCell{1}+(gamma/inPara.r(1))*eye(bits));
Y_Ie = Y_Ie*R;
Y_Ie = bsxfun(@minus, Y_Ie, median(Y_Ie, 2));
Y_Ie = Y_Ie>=0;
Bi_Ie = compactbit(Y_Ie);

TeTime2 = cputime;
testtime = TeTime2-TeIime1;