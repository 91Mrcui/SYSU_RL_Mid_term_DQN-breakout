from matplotlib import pyplot as plt
import matplotlib as mpl
def gra(path):
    y=[]
    with open(path, "r") as f:
        for line in f.readlines():
            line = list(line.strip('\n').split(" ")) #去掉列表中每一个元素的换行符
        
            while(line.count(" ")):
                line.remove(" ")
            while(line.count("")):
                line.remove("")
            #print(line)
            y.append(float(line[2]))
    return y

def avg(arry):
    #print(arry)
    res=[]
    for i in range(1,len(arry)):
        tmp=0.
        if i-5>=0:
            for j in range(i-5,i):
                tmp+=arry[j]
            tmp/=5
        else:
            for j in range(0,i):
                tmp+=arry[j]
            tmp/=(i-0)
        res.append(tmp)
    #print(res)
    return res

if __name__ == "__main__":

    natural=gra("D:\\reforce_learnig\\mid\\rewards\\natural.txt")
    prio=gra("D:\\reforce_learnig\\mid\\rewards\\prio.txt")
    prio_double=gra("D:\\reforce_learnig\\mid\\rewards\\prio_double.txt")
    prio_duel=gra("D:\\reforce_learnig\\mid\\rewards\\prio_duel.txt")
    stable_double=gra("D:\\reforce_learnig\\mid\\rewards\\stable_double.txt")
    prio_double_dueling=gra("D:\\reforce_learnig\\mid\\rewards\\prio_double_dueling.txt")
    #plt.style.use('ggplot')
    plt.style.use('seaborn-darkgrid')
    #plt.style.use('dark_background') 


    #plt.plot(natural,color='#C0C0C0',alpha=0.8)
    #plt.plot(avg(natural),color='#808A87')

    #plt.plot(prio,color="#87CEEB",alpha=0.8)
    #plt.plot(avg(prio),color="#4169E1")

    #plt.plot(prio_double,color="#DA70D6",alpha=0.8)
    #plt.plot(avg(prio_double),color="#A020F0")

    #plt.plot(prio_duel,color='#FFDEAD',alpha=0.8)
    #plt.plot(avg(prio_duel),color='#FF6100')

    plt.plot(stable_double,color='#7FFFD4',alpha=0.8)
    plt.plot(avg(stable_double),color='#03A89E')

    #plt.plot(prio_double_dueling,color='#00FF00',alpha=0.8)
    #plt.plot(avg(prio_double_dueling),color='#2E8B57')


    plt.title("Rewards of model")
    #plt.title("Avg Rewards of model")
    plt.xlabel("batches with 10000 steps")
    plt.ylabel("rewards")
    plt.grid(ls='--')
    plt.legend(["stablize+double","avg of stablize+double"])
    #plt.legend(["Prioritized","avg of Prioritized","P+double","P+dueling","stablize+double","P+dueling+double"])
    

    plt.show()