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

% �ܶ����Լ���Լ����
TonNumber = 6000;
LotSize = 5;

% Pos : ÿ��ʱ��Ӧ�ó��е�����
Pos = round(TonNumber / LotSize * pdelt);
% PosChange : ÿ����Ӧ�ý��׵����� 
PosChange = [Pos(:,1), diff(Pos,1,2)];

% HedgeProfit : ÿ��ʱ��ĶԳ�ӯ�� = ��һ��ʱ����е����� * �۸�仯 * ��Լ����
HedgeProfit = zeros(size(s));
HedgeProfit(:,2:end) = Pos(:,1:end-1) .* diff(s,1,2) * LotSize;

% CommissionFee : ���׷��� = �۸� * �������� * �����ѱ��� * ��Լ����
CommissionFee = abs(s) .* abs(PosChange) .* f .* LotSize;

% ProfitPerTon �� �����ӯ�� = (�Գ�ӯ�� - ������)/�ܶ���
ProfitPerTon = (sum(HedgeProfit,2) - sum(CommissionFee,2))/TonNumber;

% ProfitTheory ��ÿ������ӯ�� = ��������Ȩ�ں���ֵ - Ȩ����
[CP,PP] = blsprice(s0,sk,r,t,sigma,r);
ProfitTheory = max(0,sk - s(:,end)) - PP;

%%%%%%%%%% �������
% ʵ�ʶԳ��������۽���Ĳ��
D = ProfitPerTon - ProfitTheory;
[min(D), prctile(D,10), mean(D), prctile(D,90), max(D)]

% ��ͼ����
SMin = min(s(:,end));
SMax = max(s(:,end));
plot([SMin:10:SMax], max(0,sk-[SMin:10:SMax])-PP);
hold on;
scatter(s(:,end),ProfitPerTon,'r');
hold off;