clc;
clear;

x = categorical(["Low risk", "Moderately medium risk","Medium risk", "Moderately high risk", "High risk"]);
x = reordercats(x,{'Low risk' 'Moderately medium risk' 'Medium risk' 'Moderately high risk' 'High risk'});
y = [16508  8636 18841  8261 12327];
bar(x,y)

