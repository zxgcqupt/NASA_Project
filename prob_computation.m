clc;
clear;

T = 947 -401 + 1017 - 922 + 988-339 + 1034-897 + 976 - 804;
SVM_confusion = [550 82 205 69 41
    14 951 27 13 12
    153 65 592 119 59
    16 7 43 943 25
    20 17 34 22 883 ];

for i = 1:5
    SVM_confusion_norm(:,i) = SVM_confusion(:,i)./norm(SVM_confusion(:,i),1) ;
end


DNN_confusion = [567 74 204 46 56
    25 931 28 5 28
    208 72 495 97 116
    19 11 48 927 29
    38 27 45 28 838];

for i = 1:5
    DNN_confusion_norm(:,i) = DNN_confusion(:,i)./norm(DNN_confusion(:,i),1) ;
end

prop = [947-401 1017-922 988-339 1034-897 976-804]/1599;
SVM = [0.1 0.2 0.2 0.2 0.3];
DNN = [0.2 0.4 0.1 0.02 0.28];

SVM_res = SVM_confusion_norm*(prop.*SVM)';
SVM_res_norm = SVM_res/sum(SVM_res)

DNN_res = DNN_confusion_norm*(prop.*DNN)';
DNN_res_norm = DNN_res/sum(DNN_res)

