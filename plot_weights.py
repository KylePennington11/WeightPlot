import lib_formatting
import numpy as np
import matplotlib.pylab as plt
import datetime
from matplotlib.dates import date2num
from matplotlib.ticker import FuncFormatter
import sys
import glob

def filterData(xs,ys):
    listxs = list(xs)
    listys = list(ys)
    _xs = []
    _ys = []
    for i in range(len(listxs)):
        if np.isnan(listys[i]) == False:
            _xs.append(listxs[i])
            _ys.append(listys[i])

    return (_xs, _ys)

def myround(x, base=1):
    return int(base * round(float(x)/base))


def averageData(ys, window):
    out = []
    dataLen = len(ys)
    windowing = range(1, window + 1)
    for i in range(len(ys)):
        total = ys[i]
        count = 1
        for j in windowing:
            if i - j >= 0:
                total += ys[i - j]
                count += 1
            if i + j < dataLen:
                total += ys[i + j]
                count += 1
        total = total / count
        out.append(total)
    return out

def densityToFat(bodyDensity):
    return (4.57/bodyDensity - 4.142) * 100

def bodyFat(skinfold, dob, gender):
    if gender == 'Male':
        bodyDensity = 1.1093800 - 0.0008267*skinfold + 0.0000016*skinfold*skinfold - 0.0002574*(datetime.datetime.now().year-dob)
        fat = densityToFat(bodyDensity)
        return fat

    elif gender == "Female":
        return 4.03653 + 0.41563*skinfold - 0.00112*skinfold*skinfold + 0.03661*(datetime.datetime.now().year-dob)




def generateGraphs(filename):

    print ('Starting: ' + filename)

    name = 'Blank'
    gender = 'Male'
    dob = 1900
    goalWeight = 100             # kg
    fitLen = 10                 # number of point to use in fit
    projectedMonths = 4         # number of months to project
    numOfPtsToAverage = 2       # rolling average of each point
    nPtsAvgFat = 1              # rolling average of each point
    idealRate = 0.5             # kg / week


    lib_formatting.plot_params['keepAxis'].append('right')
    lib_formatting.plot_params['margin']['bottom'] = 0.23
    lib_formatting.plot_params['margin']['right'] = 0.13
    lib_formatting.plot_params['dimensions']['width'] = 800
    lib_formatting.plot_params['fontsize'] = 18
    lib_formatting.format()

    weight = []
    fat = []
    with open(filename,'r') as f:
        for line in f.readlines():
            line = line.strip('\r\n').split(',')
            if line != '':
                if len(line) == 1:
                    setting = line[0].split(':')
                    if setting[0]=='name':
                        name = setting[1]
                    elif setting[0]=='gender':
                        gender = setting[1]
                    elif setting[0]=='yearOfBirth':
                        dob = int(setting[1])
                    elif setting[0]=='goalWeight':
                        goalWeight = float(setting[1])
                    elif setting[0]=='fitLength':
                        fitLen = int(setting[1])
                    elif setting[0]=='projectedMonths':
                        projectedMonths = float(setting[1])
                    elif setting[0]=='nPtsAvgWeight':
                        numOfPtsToAverage = int(setting[1])
                    elif setting[0]=='nPtsAvgFat':
                        nPtsAvgFat = int(setting[1])
                    elif setting[0]=='idealWeightGain':
                        idealRate = float(setting[1])

                if len(line) > 1 and line[1] != '':
                    weight.append(tuple([line[0], line[1]]))

                if len(line) == 3 and line[2] != '':
                    fat.append(tuple([line[0], line[2]]))

    wstats = np.array(weight, dtype={'names':('date','weight'),'formats':('datetime64[D]','f')})
    fstats = np.array(fat, dtype={'names':('date','fat'),'formats':('datetime64[D]','f')})

    fatDates = list(map(lambda x: x.astype(datetime.datetime),fstats['date']))
    fatDates = list(map(date2num, fatDates))

    # Convert from numpy datetime64 to standard python datetime
    dates = map(lambda x: x.astype(datetime.datetime),wstats['date'])
    # Convert from datetime to matplotlib numerical representation of time
    dates = map(date2num, dates)
    dates, weights = filterData(dates,wstats['weight'])
    fdates, fats = filterData(fatDates,fstats['fat'])

    weights = averageData(weights, numOfPtsToAverage)

    L = len(dates)
    if L < fitLen:
        fitLen = L

    numDatesW = np.linspace(dates[-fitLen],datetime.date.today().toordinal()+projectedMonths*30, 100)

    fitW = np.poly1d(np.polyfit(dates[-fitLen:], weights[-fitLen:], 1))
    fitWeights = fitW(numDatesW)

    rate = fitW(datetime.date.today().toordinal()+7) - fitW(datetime.date.today().toordinal())
    goal = (goalWeight - fitW(datetime.date.today().toordinal()))*7/(rate*30)

    idealRate = np.linspace(0, projectedMonths*30/7, 20)*idealRate + fitW(datetime.date.today().toordinal())
    idealRateDates = np.linspace(datetime.date.today().toordinal(), datetime.date.today().toordinal()+projectedMonths*30, 20)

    skinfold = fstats['fat']
    skinfold = np.array(averageData(skinfold, nPtsAvgFat))

    bFat =  bodyFat(skinfold, dob, gender)

    L = len(fatDates)
    if L < fitLen:
        fitLen = L

    numDatesF = np.linspace(fatDates[-fitLen],datetime.date.today().toordinal()+projectedMonths*30, 100)
    fitF = np.poly1d(np.polyfit(fatDates[-fitLen:], bFat[-fitLen:], 1))
    fitFat = fitF(numDatesF)

    # Plot the graphs
    plt.xticks(rotation=30)
    plt.grid()

    plt.plot_date(dates, weights, label="", color="#FF2F2F", mec="#FF2F2F", ms=3)
    plt.plot_date(numDatesW, fitWeights, linestyle='--', linewidth=1, label="", color="#FF2F2F", mec="#FF2F2F", ms=0)
    plt.plot_date(idealRateDates, idealRate, linestyle=':', linewidth=1, label="", color="#FF2F2F", mec="#FF2F2F", ms=0)

    ax1 = plt.gca()
    plt.text(0.85, 0.95, "{:2.2f}".format(rate)  + ' kg/week', ha='center', va='center', transform=ax1.transAxes)
    plt.text(0.5, 0.95, "{:2.1f}".format(goal)  + ' months to goal', ha='center', va='center', transform=ax1.transAxes)

    plt.gca().yaxis.set_major_formatter(FuncFormatter(lambda x, pos: '%0.1f' % (x)))
    ax1.set_ylabel("Weight $(kg)$", color="#FF2F2F")
    ax1.set_xlabel("Date")
    #ax1.legend(loc=0, frameon=False)

    ax2 = ax1.twinx()
    ax2.plot_date(fatDates, bFat, color="#2F2FFF", mec="#2F2FFF", ms=3)
    ax2.plot_date(numDatesF, fitFat, linestyle='--', linewidth=1, label="", color="#2F2FFF", mec="#2F2FFF", ms=0)
    ax2.set_ylabel("\% Fat", color="#2F2FFF")

    
    x1,x2,y1,y2 = plt.axis()
    plt.ylim((myround(y1-5, 5), myround(y2+5, 5)))

    plt.gca().yaxis.set_major_formatter(FuncFormatter(lambda x, pos: '%0.0f' % (x)))

    plt.savefig(name + '_all.pdf',type='pdf')


    plt.xlim((datetime.date.today().toordinal()-365+projectedMonths*30, datetime.date.today().toordinal()+projectedMonths*30))

    plt.savefig(name + '_year.pdf',type='pdf')

    print ('Completed: ' + filename)
    print ()



#stats = np.loadtxt('Weight.csv', dtype={'names':('date','weight'),'formats':('datetime64[D]','f')}, delimiter=',')

for files in glob.glob("*.csv"):
    generateGraphs(files)

