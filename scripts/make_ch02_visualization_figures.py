"""2.4 시각화 절의 그림을 생성한다.

docs/ch02/visualization/*.md 에 실리는 img/*.png 를 만든다.
각 페이지의 코드 블록과 같은 코드이며, plt.show() 대신 savefig 를 쓴다.

실행:  python3 scripts/make_ch02_visualization_figures.py   (저장소 최상위에서)
필요:  numpy, pandas, matplotlib, seaborn, scipy
       문서 빌드에는 필요하지 않다. 그림은 PNG로 커밋된다.
"""

# ==================================================================
# v1.py
# ==================================================================
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

np.random.seed(0)

# 봉우리가 둘인 자료를 만든다. 0 근처 500개와 5 근처 500개를 이어 붙인다.
data_1 = np.concatenate([np.random.normal(0, 1, 500),
                         np.random.normal(5, 1, 500)])
# 비교용: 봉우리가 하나이고 평균과 퍼짐이 비슷하도록 잡은 자료
data_2 = np.random.normal(2.5, 2, 1000)

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
ax1.violinplot([data_1, data_2], showmeans=True, showmedians=True)
ax1.set_title("Violin Plot")
ax1.set_xticks([1, 2]); ax1.set_xticklabels(["Bimodal", "Unimodal"])
ax2.boxplot([data_1, data_2], labels=["Bimodal", "Unimodal"])
ax2.set_title("Box Plot")
plt.tight_layout()
plt.savefig("/Users/sungchul/Desktop/book/statistics_in_korean/docs/ch02/visualization/img/violin_vs_box_bimodal.png", dpi=140, facecolor="white")
print(f"data_1 (이봉): 평균 {data_1.mean():.2f}, 중앙값 {np.median(data_1):.2f}, 표준편차 {data_1.std():.2f}")
print(f"data_2 (단봉): 평균 {data_2.mean():.2f}, 중앙값 {np.median(data_2):.2f}, 표준편차 {data_2.std():.2f}")
q1a,q3a = np.percentile(data_1,[25,75]); q1b,q3b = np.percentile(data_2,[25,75])
print(f"data_1 IQR: [{q1a:.2f}, {q3a:.2f}]")
print(f"data_2 IQR: [{q1b:.2f}, {q3b:.2f}]")


# ==================================================================
# v2.py
# ==================================================================
import matplotlib
matplotlib.use("Agg")
import seaborn as sns, pandas as pd, matplotlib.pyplot as plt
url = "https://raw.githubusercontent.com/datasciencedojo/datasets/master/titanic.csv"
df = pd.read_csv(url)
fig, ax = plt.subplots(figsize=(10, 4))
sns.violinplot(data=df, x="Pclass", y="Age", hue="Sex", split=True, ax=ax)
ax.set_title("Age Distribution by Class and Sex (Titanic)")
plt.tight_layout()
plt.savefig("/Users/sungchul/Desktop/book/statistics_in_korean/docs/ch02/visualization/img/violin_titanic_split.png", dpi=140, facecolor="white")
print(df.groupby(["Pclass","Sex"])["Age"].agg(["count","median","mean"]).round(1).to_string())


# ==================================================================
# gc.py
# ==================================================================
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt, numpy as np, pandas as pd
from scipy import stats
OUT = "/Users/sungchul/Desktop/book/statistics_in_korean/docs/ch02/visualization/img/"

# 2. ax.plot vs ax.scatter
np.random.seed(0)
n = 10
x = stats.norm().rvs(size=n); noise = 0.7*stats.norm().rvs(size=n); y = 1+2*x+noise
fig,(a1,a2)=plt.subplots(1,2,figsize=(12,3))
ps = 100*stats.norm().rvs(size=n)**2; cv = stats.uniform().rvs(size=n)
a1.plot(x,y,'o',markersize=10,mec="red",mfc="blue",mew=3); a1.set_title("Standard Plot\nFixed Marker Size")
a2.scatter(x,y,s=ps,c=cv); a2.set_title("Scatter Plot\nVariable Marker Size")
for ax in (a1,a2):
    ax.set_xticks([]); ax.set_yticks([])
    for s in ['left','right','top','bottom']: ax.spines[s].set_visible(False)
plt.tight_layout(); plt.savefig(OUT+"gc_plot_vs_scatter.png",dpi=140,facecolor="white"); plt.close()

# 3a. simple bar
df = pd.DataFrame({'Courses':('Language','History','Geometry','Chemistry','Physics'),
                   'Number of Teachers':(7,3,9,1,2)}).set_index('Courses')
fig,ax=plt.subplots(figsize=(12,3))
ax.bar(x=range(len(df)),height=df["Number of Teachers"],tick_label=df.index,width=0.5)
ax.set_xlabel('Courses'); ax.set_ylabel('Number of Teachers'); ax.set_title("Favorite Courses of Teachers")
ax.spines['right'].set_visible(False); ax.spines['top'].set_visible(False)
plt.tight_layout(); plt.savefig(OUT+"gc_bar_simple.png",dpi=140,facecolor="white"); plt.close()

# 3b. grouped bar
df2 = pd.DataFrame({'Student':['Brandon','Vanessa','Daniel','Kevin','William'],
                    'Midterm':[85,60,60,65,100],'Final':[90,90,65,80,95]}).set_index('Student')
pos=np.arange(len(df2)); w=0.3
fig,ax=plt.subplots(figsize=(12,3))
ax.bar(pos-w/2,df2['Midterm'],width=w,label="Midterm"); ax.bar(pos+w/2,df2['Final'],width=w,label="Final")
ax.set_xticks(pos); ax.set_xticklabels(df2.index); ax.set_xlabel("Student"); ax.set_ylabel("Scores")
ax.set_title("Midterm and Final Scores"); ax.legend(title="Exam Type")
ax.spines['right'].set_visible(False); ax.spines['top'].set_visible(False)
plt.tight_layout(); plt.savefig(OUT+"gc_bar_grouped.png",dpi=140,facecolor="white"); plt.close()

# 3c. stacked bar
labels=("Yes","No"); counts=(np.array([95,90,40]),np.array([5,10,60])); ages=("Adults","Children","Infants")
fig,ax=plt.subplots(figsize=(6,3)); bottom=np.zeros(3)
for lab,c in zip(labels,counts):
    ax.bar(np.arange(3),c,width=0.5,bottom=bottom,tick_label=ages,label=lab); bottom+=c
ax.set_title("Has Antibodies?"); ax.spines['top'].set_visible(False); ax.spines['right'].set_visible(False)
ax.legend(title="Response",loc="center left",bbox_to_anchor=(1.0,0.5))
plt.tight_layout(); plt.savefig(OUT+"gc_bar_stacked.png",dpi=140,facecolor="white"); plt.close()

# 4. pie
fig,ax=plt.subplots()
ax.pie([215,130,245,210],explode=(0.1,0,0,0),labels=('Apples','Bananas','Cherries','Dates'),
       colors=['gold','yellowgreen','lightcoral','lightskyblue'],autopct='%1.1f%%',shadow=True,
       startangle=140,radius=1.5,counterclock=True)
ax.axis('equal'); ax.set_title('Fruit Distribution in Basket')
plt.tight_layout(); plt.savefig(OUT+"gc_pie.png",dpi=140,facecolor="white",bbox_inches="tight"); plt.close()

# 7. dot plot
data=[5,7,5,9,7,7,6,9,9,9,10,12,12,7]
freq={}
for a in data: freq[a]=freq.get(a,0)+1
fig,ax=plt.subplots(figsize=(12,3))
for a,f in freq.items(): ax.plot([a]*f,range(1,f+1),'ok')
ax.set_xlabel('Ages'); ax.set_ylabel('Number of Students'); ax.set_title("Ages of Students in Class")
ax.spines['top'].set_visible(False); ax.spines['right'].set_visible(False)
ax.set_yticks([0,1,2,3,4]); ax.spines["bottom"].set_position("zero")
plt.tight_layout(); plt.savefig(OUT+"gc_dotplot.png",dpi=140,facecolor="white"); plt.close()
print("saved 6 figures")


# ==================================================================
# gc2.py
# ==================================================================
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt, pandas as pd, seaborn as sns
OUT = "/Users/sungchul/Desktop/book/statistics_in_korean/docs/ch02/visualization/img/"
# 5. pairplot
url="https://raw.githubusercontent.com/datasciencedojo/datasets/master/titanic.csv"
df=pd.read_csv(url,index_col='PassengerId')
df['Sex_int']=df['Sex'].apply(lambda x: 1 if x=='male' else 0)
g=sns.pairplot(df[["Survived","Age","Sex_int"]])
g.figure.suptitle("Pair Plot: Survived, Age, Sex", y=1.02)
g.savefig(OUT+"gc_pairplot.png",dpi=120,facecolor="white",bbox_inches="tight")
print("pairplot saved")

# 8. crosstab (text output)
data={'SUV':28*['yes']+35*['no']+97*['yes']+104*['no'],
      'Accident':28*['yes']+35*['yes']+97*['no']+104*['no']}
d=pd.DataFrame(data)
dg=pd.crosstab(d.SUV,d.Accident,rownames=['SUV'],colnames=['Accident'])
dg.loc['TOTAL',:]=dg.sum(); dg.loc[:,'TOTAL']=dg.sum(axis=1); dg=dg.astype(int)
print("=== 도수분포표 ==="); print(dg)
dh=dg/dg.loc['TOTAL','TOTAL']
print("=== 상대도수분포표 ==="); print(dh)


# ==================================================================
# gc4.py
# ==================================================================
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt, numpy as np, pandas as pd
OUT="/Users/sungchul/Desktop/book/statistics_in_korean/docs/ch02/visualization/img/"
rng = np.random.default_rng(42)
dates = pd.bdate_range("2023-01-01", "2023-12-31")
returns = rng.normal(0.0004, 0.02, len(dates))
price = 20000 * np.exp(np.cumsum(returns))
s = pd.Series(price, index=dates, name="Close")
fig, ax = plt.subplots(figsize=(12, 3))
ax.plot(s.index, s.values, color="blue", lw=1.2, label="Close")
mark = pd.Timestamp("2023-10-19")
ax.plot([mark], [s.loc[mark]], "or", ms=8, label=f"{mark.date()}: {s.loc[mark]:,.0f}")
ax.set_xlabel("Date"); ax.set_ylabel("Price (KRW)")
ax.set_title("Simulated Daily Closing Price, 2023")
ax.legend(); ax.spines[['top','right']].set_visible(False)
plt.tight_layout(); plt.savefig(OUT+"gc_lineplot_timeseries.png",dpi=140,facecolor="white")
print(f"거래일 수: {len(s)}")
print(f"기간: {s.index.min().date()} ~ {s.index.max().date()}")
print(f"시작가 {s.iloc[0]:,.0f}  종료가 {s.iloc[-1]:,.0f}")
print(f"최저 {s.min():,.0f}  최고 {s.max():,.0f}")
print(f"2023-10-19 종가: {s.loc[mark]:,.0f}")

# 줄기잎그림 (stemgraphic 없이 직접 구현)
def stem_leaf(data, stem_unit=10):
    from collections import defaultdict
    d = defaultdict(list)
    for v in sorted(data):
        d[int(v)//stem_unit].append(int(v) % stem_unit)
    lines = ["줄기 | 잎", "-----+" + "-"*20]
    for st in range(min(d), max(d)+1):
        leaves = " ".join(str(l) for l in d.get(st, []))
        lines.append(f"{st:4d} | {leaves}")
    lines.append(f"\n줄기 단위 = {stem_unit}   (줄기 6, 잎 5  ->  65)")
    return "\n".join(lines)

scores = [65, 93, 45, 73, 99, 70, 88, 46, 75, 34, 83, 100, 88, 72, 70]
print("\n=== 줄기잎그림 ===")
print(stem_leaf(scores))


# ==================================================================
# gce.py
# ==================================================================
import matplotlib; matplotlib.use("Agg")
import pandas as pd, numpy as np, matplotlib.pyplot as plt, seaborn as sns
OUT="/Users/sungchul/Desktop/book/statistics_in_korean/docs/ch02/visualization/img/"

# 예제 1
np.random.seed(42)
airlines=['American','Delta','Southwest','United']; n_obs=100
params={'American':(2,3),'Delta':(1.5,2.5),'Southwest':(1.2,2),'United':(1.8,3.2)}
rows=[]
for a in airlines:
    sh,sc=params[a]
    for d in np.random.gamma(shape=sh,scale=sc,size=n_obs): rows.append({'airline':a,'pct_carrier_delay':d})
airline_stats=pd.DataFrame(rows)
print("=== 예제 1: 항공사별 지연 요약 ===")
print(airline_stats.groupby('airline')['pct_carrier_delay'].describe()[['count','mean','50%','std','max']].round(2).to_string())

fig,ax=plt.subplots(figsize=(8,5))
airline_stats.boxplot(by='airline',column='pct_carrier_delay',ax=ax)
ax.set_xlabel('Airline'); ax.set_ylabel('Daily % of Delayed Flights')
ax.set_title('Airline Delay Comparison: Boxplots'); plt.suptitle('')
plt.tight_layout(); plt.savefig(OUT+"gce_airline_box.png",dpi=140,facecolor="white"); plt.close()

fig,ax=plt.subplots(figsize=(8,5))
sns.violinplot(data=airline_stats,x='airline',y='pct_carrier_delay',ax=ax,inner='quartile',color='lightblue')
ax.set_xlabel('Airline'); ax.set_ylabel('Daily % of Delayed Flights')
ax.set_title('Airline Delay Comparison: Violin Plots')
plt.tight_layout(); plt.savefig(OUT+"gce_airline_violin.png",dpi=140,facecolor="white"); plt.close()

# 예제 2
np.random.seed(123)
zips=[98188,98105,98108,98126]; n_homes=150; rows=[]
for z in zips:
    base=300_000 if z in [98105,98108] else 450_000
    pr=np.clip(np.random.normal(base,100_000,n_homes),50_000,2_000_000)
    for p in pr: rows.append({'ZipCode':str(z),'TaxAssessedValue':p})
housing=pd.DataFrame(rows)
print("\n=== 예제 2: 우편번호별 주택가치 (천 달러) ===")
print((housing.groupby('ZipCode')['TaxAssessedValue'].agg(['count','mean','median','std'])/1000).round(1).assign(count=lambda d:(d['count']*1000).astype(int)).to_string())
fig,ax=plt.subplots(figsize=(8,5))
housing.boxplot(by='ZipCode',column='TaxAssessedValue',ax=ax)
ax.set_xlabel('Zip Code'); ax.set_ylabel('Tax Assessed Value ($)')
ax.set_title('Housing Values Across Neighborhoods'); plt.suptitle('')
plt.tight_layout(); plt.savefig(OUT+"gce_housing_box.png",dpi=140,facecolor="white"); plt.close()

# 예제 3
np.random.seed(456)
grades=list('ABCDEFG'); rows=[]
for g in grades:
    i=ord(g)-ord('A')
    inc=np.clip(np.random.normal(80_000-i*8_000,15_000+i*5_000,100),10_000,200_000)
    for v in inc: rows.append({'grade':g,'income':v})
loans=pd.DataFrame(rows)
print("\n=== 예제 3: 신용등급별 소득 (천 달러) ===")
print((loans.groupby('grade')['income'].agg(['mean','median','std'])/1000).round(1).to_string())
fig,ax=plt.subplots(figsize=(10,5))
sns.violinplot(data=loans,x='grade',y='income',ax=ax,color='lightgreen')
ax.set_xlabel('Loan Grade (A=best, G=worst)'); ax.set_ylabel('Annual Income ($)')
ax.set_title('Income Distribution by Credit Grade')
plt.tight_layout(); plt.savefig(OUT+"gce_loans_violin.png",dpi=140,facecolor="white"); plt.close()
print("\nsaved 4 figures")
