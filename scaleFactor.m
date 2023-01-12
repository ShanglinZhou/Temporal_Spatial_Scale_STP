function [SSI, TSF, SSF] = scaleFactor_mean_sz(x1,x2,tLimit,sLimit,step)
[N,T1] = size(x1);
T2 = size(x2,2);


if T1 > T2
    temp = x1;
    x1 = x2;
    x2 = temp;
    
    temp = T1;
    T1  =T2;
    T2 = temp;
    
    SWITCH = 1;
else
    SWITCH = 0;
    
end

tFac = tLimit(1):step:tLimit(2);
sFac = sLimit(1):step:sLimit(2);

numT = length(tFac);
numS = length(sFac);

Dist = zeros(numT,numS);
Corr = zeros(numT,numS);

T_all = floor(tLimit(2)*T1);
% y2 = zeros(N,T_all);
y2 = repmat(mean(x2,2),1,T_all);
y2(:,1:T2) = x2;


for tInd = 1:numT    
    T1_warp = floor(T1*tFac(tInd));
   for sInd = 1:numS
%        y1 = zeros(N,T_all);
       
       temp = x1';
       temp = interp1(temp,linspace(1,T1,T1_warp))';
       temp = temp*sFac(sInd);
       
       y1 = repmat(mean(temp,2),1,T_all);
%        y1 = repmat(mean(x2,2),1,T_all);
       y1(:,1:T1_warp) = temp;
       
%        Dist(tInd,sInd) = mean(vecnorm(y1-y2));
       T_measure = max(T2,T1_warp);
       normm = mean(vecnorm(y2(:,1:T_measure)-mean(y2(:,1:T_measure),2)));
       Dist(tInd,sInd) = mean(vecnorm(y1(:,1:T_measure)-y2(:,1:T_measure)))./normm;
%        Corr(tInd,sInd) = corr(y1(:),y2(:));
   end
end

[minDist,Ind] = min(Dist,[],'all','linear');
[r,l] =  ind2sub(size(Dist),Ind);

% SSI = 1 - Corr(r,l);
SSI = minDist;
if SWITCH
    TSF = 1/tFac(r);
    SSF = 1/sFac(l);
else
    TSF = tFac(r);
    SSF = sFac(l);
end

if sum(TSF==tLimit) || sum(SSF==sLimit) 
    warning('Searching reaches the limit, chage the searching limit or scaling profile is not applicable to this case.')
end

end