% Reference:
% Di Wang, Xinbo Gao, Xiumei Wang, and Lihuo He. 
% Semantic Topic Multimodal Hashing for Cross-Media Retrieval. 
% Proceedings of the Twenty-Fourth International Joint Conference on Artificial Intelligence, 3890-3896, Buenos Aires, Argentina, 25¨C31 July 2015.
% (Manuscript)
%
% Contant: Di Wang (wangdi@xidain.edu.cn)
%
clc;clear 
load wikiData.mat
%% Calculate the groundtruth
GT = L_te*L_tr';
WtrueTestTraining = zeros(size(L_te,1),size(L_tr,1));
WtrueTestTraining(GT>0)=1;
%% Parameter setting
bit = 32; 
%% Learn STMH
[B_I,B_T,tB_I,tB_T] = main_STMH(I_tr, T_tr, I_te, T_te, bit);
%% Compute mAP
Dhamm = hammingDist(tB_I, B_T)';    
[~, HammingRank]=sort(Dhamm,1);
mapIT = map_rank(L_tr,L_te,HammingRank); 
Dhamm = hammingDist(tB_T, B_I)';    
[~, HammingRank]=sort(Dhamm,1);
mapTI = map_rank(L_tr,L_te,HammingRank); 
map = [mapIT(100),mapTI(100)];
