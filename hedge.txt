s0 = 41350;
npath = 10000;
nstep = 252;
sigma = 0.2;
r = 0.05;
sk = 35000;
f = 0.0001;
t = 1;
rx = randn(npath,nstep);
deltat = t/nstep;
s = [s0*ones(npath,1) zeros(npath,nstep-1)];
for i=1:nstep-1
    s(:,i+1) = s(:,i) + s(:,i)*r*deltat+sigma*deltat^0.5*(s(:,i).*rx(:,i));
end

for m = 1:nstep
    [cdelt(:,m),pdelt(:,m)]=blsdelta(s(:,m),sk,r,1-(m-1)*deltat,sigma,0);
end;

% 总吨数以及合约乘数
TonNumber = 6000;
LotSize = 5;

% Pos : 每个时点应该持有的手数
Pos = round(TonNumber / LotSize * pdelt);
% PosChange : 每个点应该交易的手数 
PosChange = [Pos(:,1), diff(Pos,1,2)];

% HedgeProfit : 每个时点的对冲盈亏 = 上一个时点持有的手数 * 价格变化 * 合约乘数
HedgeProfit = zeros(size(s));
HedgeProfit(:,2:end) = Pos(:,1:end-1) .* diff(s,1,2) * LotSize;

% CommissionFee : 交易费用 = 价格 * 交易手数 * 手续费比例 * 合约乘数
CommissionFee = abs(s) .* abs(PosChange) .* f .* LotSize;

% ProfitPerTon ： 最后总盈亏 = (对冲盈亏 - 手续费)/总吨数
ProfitPerTon = (sum(HedgeProfit,2) - sum(CommissionFee,2))/TonNumber;

% ProfitTheory ：每吨理论盈亏 = 到期日期权内含价值 - 权利金
[CP,PP] = blsprice(s0,sk,r,t,sigma,r);
ProfitTheory = max(0,sk - s(:,end)) - PP;

%%%%%%%%%% 结果检验
% 实际对冲结果和理论结果的差别
D = ProfitPerTon - ProfitTheory;
[min(D), prctile(D,10), mean(D), prctile(D,90), max(D)]

% 画图检验
SMin = min(s(:,end));
SMax = max(s(:,end));
plot([SMin:10:SMax], max(0,sk-[SMin:10:SMax])-PP);
hold on;
scatter(s(:,end),ProfitPerTon,'r');
hold off;