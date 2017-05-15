import numpy as np
import pandas as pd    
import rdflib
from rdflib import Namespace
import math
import scipy
from scipy import stats
import random
import matplotlib.pyplot as plt
import multipolyfit as mpf



def remove_duplicates(seq):
    seen = set()
    seen_add = seen.add
    return [x for x in seq if not (x in seen or seen_add(x))]



def findCols(a):
    colNames = list(a)
    if str(colNames[0]) == 'work id':
        xcol = str(colNames[1])
        ycol = str(colNames[2])
    elif str(colNames[1]) == 'work id':
        xcol = str(colNames[0])
        ycol = str(colNames[2])
    if str(list(a)[2]) == 'work id':
        xcol = str(colNames[0])
        ycol = str(colNames[1])
        
    return xcol, ycol



def graph_compareData(data_x, data_y, zoom=(0,0), over_x=pd.DataFrame(), over_y=pd.DataFrame(), title='', highlight=list(), decrease_alpha=False, best_fit=False):
    
    if over_x.empty and over_y.empty:
        a = pd.merge(data_x, data_y, on='work id')
        
    else:
        if not over_x.empty:
            b = pd.merge(data_x, over_x, on='work id')
            xcol, ycol = findCols(b)
            b[xcol] = pd.to_numeric(b[xcol])
            b[ycol] = pd.to_numeric(b[ycol])
            
            resultColumn = xcol + """
            / """ + ycol
            b[resultColumn] = b[xcol].div(b[ycol])
            b = b[['work id', resultColumn]]
        if not over_y.empty:
            c = pd.merge(data_y, over_y, on='work id')
            xcol, ycol = findCols(c)
            c[xcol] = pd.to_numeric(c[xcol])
            c[ycol] = pd.to_numeric(c[ycol])
            
            resultColumn = xcol + """
            / """ + ycol
            c[resultColumn] = c[xcol].div(c[ycol])
            c = c[['work id', resultColumn]]
            
        if not over_x.empty and not over_y.empty:
            a = pd.merge(b, c, on='work id')
        elif not over_x.empty and over_y.empty:
            a = pd.merge(b, data_y, on='work id')
        elif over_x.empty and not over_y.empty:
            a = pd.merge(data_x, c, on='work id')

            
    xcol, ycol = findCols(a)
    
    
    df = a[[xcol, ycol]]
    x = list(map(float, list(df[xcol])))
    y = list(map(float, list(df[ycol])))

    if len(highlight) > 0:
        a1 = a.loc[a['work id'].isin(highlight)]
        a2 = a.loc[~a['work id'].isin(highlight)]

        df1 = a1[[xcol, ycol]]
        x1 = list(map(float, list(df1[xcol])))
        y1 = list(map(float, list(df1[ycol])))

        df2 = a2[[xcol, ycol]]
        x2 = list(map(float, list(df2[xcol])))
        y2 = list(map(float, list(df2[ycol])))
    else:
        x2 = x
        y2 = y


    if zoom != (0,0):
        plt.axis([0, zoom[0], 0, zoom[1]])


    plt.xlabel(xcol)
    plt.ylabel(ycol)

    if decrease_alpha:
        plt.scatter(x2, y2, s=20, alpha=0.05, color='red')
    else:
        plt.scatter(x2, y2, s=20, alpha=0.2, color='red')
        plt.scatter(x2, y2, s=20, alpha=0.5, facecolors='none', edgecolors='black')

    if len(highlight) > 0:
        plt.scatter(x1, y1, s=50, alpha=1, marker='x', color='blue')

    plt.title(title)

    if best_fit == True:
        return plt.plot(np.unique(x), np.poly1d(np.polyfit(x, y, 1))(np.unique(x)))
    else:
        return plt.plot()



def table_worksByDate(g,collection):
    worksByDate = pd.DataFrame()
    workFrequency = pd.DataFrame()
    
    for work in collection:
        # First, the main RDF query to obtain relevant info about overlapping texts and their dates.
        result = g.query(
            """
            PREFIX astr: <http://www.astronomoumenos.com/ontologies/astr.owl#>
            PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#>
            SELECT ?wo1 ?lw ?la (MIN(?d1) AS ?date1) (MAX(?d2) AS ?date2) WHERE {
                BIND( <http://www.astronomoumenos.com/id/%s> AS ?wo1 ) .
                <http://www.astronomoumenos.com/id/%s> rdfs:label ?lw .
                <http://www.astronomoumenos.com/id/%s> astr:hasContributor ?au .
                ?au rdfs:label ?la .
                ?wi astr:witnessOf <http://www.astronomoumenos.com/id/%s> .
                ?wi astr:hasClaim/astr:hasStartDate ?d1 .
                ?wi astr:hasClaim/astr:hasEndDate ?d2 .
                }
                GROUP BY ?wi
                """ % (work, work, work, work))
        worksByDate = worksByDate.append(result.bindings)
        
    # There is a chance that the above query has no results.
    for work in collection:
        
        # Next, the secondary query that will be used to determine how many times each pair appears.
        result = g.query(
            """
            PREFIX astr: <http://www.astronomoumenos.com/ontologies/astr.owl#>
            PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#>
            PREFIX rdf: <http://www.w3.org/1999/02/22-rdf-syntax-ns#>
            SELECT ?wo1 (COUNT(?ms) AS ?c) WHERE {
                BIND( <http://www.astronomoumenos.com/id/%s> AS ?wo1 )
                ?ms astr:msContains ?wi1 .
                ?wi1 astr:witnessOf <http://www.astronomoumenos.com/id/%s> .
                }
                GROUP BY ?wo1
                """ % (work, work))
        workFrequency = workFrequency.append(result.bindings)
        
    a = pd.merge(worksByDate, workFrequency, on=rdflib.term.Variable('wo1'))
            
    # The data is sorted for ease of reading on the output table or optional graph.
    worksByDate_sorted = a.sort_values(rdflib.term.Variable('c'), ascending=False).reset_index(drop=True)

    return worksByDate_sorted
                
def graph_worksByDate(worksByDate_sorted,title=""):
    '''
    '''
    graphTitle = title
    # It is possible that no pandas dataframe will be returned by `table_neighborsByDate()`. This if statement catches that case.
    if str(type(worksByDate_sorted)) == "<class 'NoneType'>":
        return None
    
    # It is possible that an empty dataframe will be returned by `table_neighborsByDate()`. This if statement catches that case.
    elif worksByDate_sorted.empty:
        return None
    
    # Otherwise the function proceeds as normal.
    else:
        # Note that the following arrays are very often aligned so that list1[i] == list2[i].
        
        # Obtain an array of start dates for each of the date ranges. To provide in the graph a visual distinction between two mss (10th c and 11th c) and one ms (10-11th c), 1 is added to the start date.
        start = list(map(int, list(worksByDate_sorted[rdflib.term.Variable('date1')])))
        start[:] = [x + 1 for x in start]
        start = np.array(start)
        
        # Obtain an array of end dates for each of the date ranges. To provide in the graph a visual distinction between two mss (10th c and 11th c) and one ms (10-11th c), 1 is subracted from the end date.
        end = list(map(int, list(worksByDate_sorted[rdflib.term.Variable('date2')])))
        end[:] = [x - 1 for x in end]
        end = np.array(end)
        
        # The span of each date range. Exact dates have not yet been added but will likely span one year.
        width = end-start
        
        # The full arrays contain repeats of works because multiple witnesses are plotted on the graph. However, unique lists must be provided to create unique yticks in the chart.
        titles_all = list(map(str, list(worksByDate_sorted[rdflib.term.Variable('lw')])))
        authors_all = list(map(str, list(worksByDate_sorted[rdflib.term.Variable('la')])))
        count_all = list(map(str, list(worksByDate_sorted[rdflib.term.Variable('c')])))
        
        i = 0
        for title in titles_all:
            titles_all[i] = authors_all[i] + " " + title + " (" + count_all[i] + " mss)"
            i = i + 1
        yval_all = np.arange(len(titles_all))
        titles = remove_duplicates(titles_all)
        yval = np.arange(len(remove_duplicates(titles)))

        titles_vals = dict()
        for i in range (0, len(yval)):
            titles_vals[titles[i]] = yval[i]

        # Graphs will vary in height according to how many overlapping texts are included.
        num = int( float(len(titles)) / float(2.5) ) + 1
        plt.rcParams["figure.figsize"] = (8,num)

        # Plotting date ranges on the graph. An empty array is first provided to set up the base yticks.
        fig, ax = plt.subplots()
        for val in yval:
            ax.barh(bottom=val, width=0, left=min(start), height=0.5, alpha=0.1, color='#000000',) # min(start) because otherwise graph is moved over to start at 0
        for i in range(0, len(titles_all)):
            ax.barh(bottom=titles_vals[titles_all[i]], width=width[i], left=start[i], height=0.5, alpha=0.05, color='#000000',)   

        # Graph info
        plt.yticks(yval, titles)
        plt.xlabel('Manuscript Date')
        plt.title(graphTitle)

        return plt.show()


def table_neighborsByDate(g,work,n=1):
    '''
    Returns a pandas dataframe containing data regarding overlapping texts in the manuscripts,
     their frequency, and the date ranges for each of the witnesses. Intended for use with
     `graph_neighborsByDate()`, which graphs the results in a bar chart.

    Parameters:
    -----------
      g : An rdflib RDF graph containing the relevant astr database.
      work : The astr ID of the work in question (astrID_wo). Follows the pattern wo#######.
      n : Optional; defaults to 1. An integer value -- if pairs of texts overlap in this many
       or fewer manuscripts, then those results are dropped from the table that will be returned.
       Set up to prevent the long tail of pairs that overlap in a single manuscript from crowding
       the table / optional `graph_neighborsByDate()` graph. If the user is not interested in
       pairs occuring in fewer than for example 10 manuscripts, the appropriate number may be
       entered for n.

    Returns:
    --------
      neighborsByDate_dropped : A pandas dataframe. Contains key information such as the title of
       the input text (title), the overlapping text's identifier (wo2), title (lw), and author (la),
       the starting date for the date range given to the manuscript witness (date1), and the ending 
       date for the date range given to the manuscript witness (date2). Any pairs which occur in
       fewer than n manuscripts have been dropped.
    '''
    
    # First, the main RDF query to obtain relevant info about overlapping texts and their dates.
    result = g.query(
        """
        PREFIX astr: <http://www.astronomoumenos.com/ontologies/astr.owl#>
        PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#>
        SELECT ?title ?wo2 ?lw ?la (MIN(?d1) AS ?date1) (MAX(?d2) AS ?date2) WHERE {
            ?ms astr:msContains ?wi1 .
            ?wi1 astr:witnessOf <http://www.astronomoumenos.com/id/%s> .
            <http://www.astronomoumenos.com/id/%s> rdfs:label ?title .
            ?ms astr:msContains ?wi2 .
            ?wi2 astr:witnessOf ?wo2 .
            ?wo2 rdfs:label ?lw .
            ?wo2 astr:hasContributor ?au .
            ?au rdfs:label ?la .
            ?wi2 astr:hasClaim/astr:hasStartDate ?d1 .
            ?wi2 astr:hasClaim/astr:hasEndDate ?d2 .
            FILTER( ?wo2 != <http://www.astronomoumenos.com/id/%s> )
            }
            GROUP BY ?wi2
            """ % (work, work, work))
    neighborsByDate = pd.DataFrame(result.bindings)
    
    # There is a chance that the above query has no results.
    if neighborsByDate.empty:
        return None
    
    else:
        # Next, the secondary query that will be used to determine how many times each pair appears.
        result = g.query(
            """
            PREFIX astr: <http://www.astronomoumenos.com/ontologies/astr.owl#>
            PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#>
            PREFIX rdf: <http://www.w3.org/1999/02/22-rdf-syntax-ns#>
            SELECT ?wo2 ?ms ?lw ?la WHERE {
                ?ms astr:msContains ?wi1 .
                ?wi1 astr:witnessOf <http://www.astronomoumenos.com/id/%s> .
                ?ms astr:msContains ?wi2 .
                ?wi2 astr:witnessOf ?wo2 .
                ?wo2 rdfs:label ?lw .
                ?wo2 astr:hasContributor ?au .
                ?au rdfs:label ?la .
                FILTER( ?wo2 != <http://www.astronomoumenos.com/id/%s> )
                }
                GROUP BY ?wi2
                """ % (work, work))
        neighborFrequency = pd.DataFrame(result.bindings)
        
        # There is a chance that the above query has no results.
        if neighborFrequency.empty:
            return None
        
        else:
            # A dictionary, dictDF, is built to pair the astr work ID (wo2) of the overlapping text with how many times it occurs.
            dictDF = dict()
            for index, row in neighborFrequency.iterrows():
                author_title = row[rdflib.term.Variable('wo2')]
                if author_title not in dictDF:
                    dictDF[author_title] = list()
                    dictDF[author_title].append(1)
                else:
                    dictDF[author_title][0] = dictDF[author_title][0] + 1

            # Any works which overlap a number of times below the optionally input threshold are removed.
            delete = list()
            for key in dictDF:
                if dictDF[key][0] <= n:
                    delete.append(key)

            multWo = list()
            for key in delete:
                del dictDF[key]

            for key in dictDF:
                multWo.append(key)

            neighborsByDate_dropped = neighborsByDate.loc[neighborsByDate[rdflib.term.Variable('wo2')].isin(multWo)].reset_index(drop=True)

            # It is possible that dropping overlapping works with a frequency below the threshold results in an empty dataframe.
            if neighborsByDate_dropped.empty:
                return None
            
            else:
                # The frequency data is added to the table output from the first SPARQL query.
                neighborsByDate_dropped['c'] = 0
                i = 0
                for index, row in neighborsByDate_dropped.iterrows():
                    neighborsByDate_dropped.set_value(i,'c',dictDF[row[rdflib.term.Variable('wo2')]][0])
                    i = i + 1

                # The data is sorted for ease of reading on the output table or optional graph.
                neighborsByDate_dropped = neighborsByDate_dropped.sort_values('c', ascending=False).reset_index(drop=True)

                return neighborsByDate_dropped

def graph_neighborsByDate(neighborsByDate_dropped):
    '''
    Returns a matplotlib pyplot as a bar chart depicting which texts overlap with the given
    text in manuscripts over time. Takes as input a pandas dataframe produced by 
    `table_neighborsByDate()`. The x-axis denotes time and the y-axis presents the authors
    and titles of each overlapping work presented -- it is noted in parentheses in how many
    manuscripts the input text and the overlapping text appear together. Transparent bars
    are used so that if multiple manuscripts are represented from the 12th century, for
    instance, the darkness of the bar can indicate this higher number.

    Parameters:
    -----------
      neighborsByDate_dropped : A pandas dataframe output by `table_neighborsByDate()`.
       Contains key information such as the title of the input text (title), the overlapping
       text's identifier (wo2), title (lw), and author (la), the starting date for the date
       range given to the manuscript witness (date1), and the ending date for the date range
       given to the manuscript witness (date2).

    Returns:
    --------
      plt.show() : The graph to be displayed.
    '''
    
    # It is possible that no pandas dataframe will be returned by `table_neighborsByDate()`. This if statement catches that case.
    if str(type(neighborsByDate_dropped)) == "<class 'NoneType'>":
        return None
    
    # It is possible that an empty dataframe will be returned by `table_neighborsByDate()`. This if statement catches that case.
    elif neighborsByDate_dropped.empty:
        return None
    
    # Otherwise the function proceeds as normal.
    else:
        # Note that the following arrays are very often aligned so that list1[i] == list2[i].
        
        # Obtain an array of start dates for each of the date ranges. To provide in the graph a visual distinction between two mss (10th c and 11th c) and one ms (10-11th c), 1 is added to the start date.
        start = list(map(int, list(neighborsByDate_dropped[rdflib.term.Variable('date1')])))
        start[:] = [x + 1 for x in start]
        start = np.array(start)
        
        # Obtain an array of end dates for each of the date ranges. To provide in the graph a visual distinction between two mss (10th c and 11th c) and one ms (10-11th c), 1 is subracted from the end date.
        end = list(map(int, list(neighborsByDate_dropped[rdflib.term.Variable('date2')])))
        end[:] = [x - 1 for x in end]
        end = np.array(end)
        
        # The span of each date range. Exact dates have not yet been added but will likely span one year.
        width = end-start
        
        # The full arrays contain repeats of works because multiple witnesses are plotted on the graph. However, unique lists must be provided to create unique yticks in the chart.
        titles_all = list(map(str, list(neighborsByDate_dropped[rdflib.term.Variable('lw')])))
        authors_all = list(map(str, list(neighborsByDate_dropped[rdflib.term.Variable('la')])))
        count_all = list(map(str, list(neighborsByDate_dropped['c'])))
        i = 0
        for title in titles_all:
            titles_all[i] = authors_all[i] + " " + title + " (" + count_all[i] + " mss)"
            i = i + 1
        yval_all = np.arange(len(titles_all))
        titles = remove_duplicates(titles_all)
        yval = np.arange(len(remove_duplicates(titles)))

        titles_vals = dict()
        for i in range (0, len(yval)):
            titles_vals[titles[i]] = yval[i]

        # Graphs will vary in height according to how many overlapping texts are included.
        num = int( float(len(titles)) / float(2.5) ) + 1
        plt.rcParams["figure.figsize"] = (8,num)

        # Plotting date ranges on the graph. An empty array is first provided to set up the base yticks.
        fig, ax = plt.subplots()
        for val in yval:
            ax.barh(bottom=val, width=0, left=min(start), height=0.5, alpha=0.1, color='#000000',) # min(start) because otherwise graph is moved over to start at 0
        for i in range(0, len(titles_all)):
            ax.barh(bottom=titles_vals[titles_all[i]], width=width[i], left=start[i], height=0.5, alpha=0.05, color='#000000',)   

        # Graph info
        plt.yticks(yval, titles)
        plt.xlabel('Manuscript Date')
        plt.title('Neighbors of \'' + str(neighborsByDate_dropped[rdflib.term.Variable('title')][0]) + '\' by manuscript date')

        return plt.show()