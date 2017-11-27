#### 数据处理原理
假设数据块长度为DT，起始时刻为-DT/2，结束时刻为DT/2，将相位在数据块中点(0点)作泰勒展开(3阶)：
Phi(t)=c0+c1*t+c2*t^2+c3*t^3
则DT时段内信号可以表示为：
s(t)=(c4+c5*t)cos(Phi(t)) (c4.c5为信号幅度参数)
通过拟合的办法可以得到c0~c5六个参数

#### 多普勒输出说明
每个数据块有三行输出，一共13个变量，各变量定义如下(除去时间)：
   --  var1    var2
b  |
l  |
o  --  var3    var4   var5    var6   var7    var8   var9    var10  var11 
c  |
k  |
   --  var12   var13

var1	:数据块开始时刻实时相位mod(Phi(-DT/2)/2pi) 单位:rad
var2	:数据块开始时刻实时频率f=c1+2*c2*t+3*c3*t^2|t=-DT/2 单位：rad/s
var3	:c0
var4	:c1
var5	:c2
var6	:c3
var7	:c4
var8	:c5
var9	:数据块的积分相位(Phi(DT/2)-Phi(-DT/2))
var10	:RSS(拟后残差)
var11	:数据质量控制字(1/0:可用/不可用)
var12	:数据块结束时刻实时相位mod(Phi(DT/2)/2pi) 单位:rad
var13	:数据块结束时刻实时频率f(DT/2)=c1+2*c2*t+3*c3*t^2|t=Dt/2  单位：rad/s


####	数据检核
1，通过相邻数据块实时相位和频率的连续性可以检核数据的可靠性
2，通过RSS的分布来检核
 
