#!/usr/bin/env python3

"""A pathetically simple Python script to convert (my) simple graph representations
to a dot(1) configuration file.  Keith Hellman <khellman@mines.edu> for 
Spring 2014, 2016 Compilers.

On a Linux box with dot(1) installed (typically found in the graphvis package),
invoke like this:

 $ ls
 parsetree.txt
 $ cat parsetree.txt | ./tree-to-graphvis | dot -Tpng -o parsetree.png
 $ see parsetree.png

or

 $ display parsetree.png

Input format is in two parts:  node identification and then edge identification.

 $ cat parsetree.txt
 nodeA Node A . shape=box
 leafB Leaf B
 nodeC Node C
 leafD Leaf D
 leafE Leaf E
 
 nodeA leafB nodeC
 nodeC leafD leafE
 $ 

This would generate a graph:
  +--------+
  | Node A |
  +--------+
    /    \
 Leaf B  Node C
         /   \
     Leaf D   \
             Leaf E


The input format is line oriented, ignores empty lines, and does not support
comments.  

='d keypairs following . can be used for dot attributes; see dot language 
definition for possible attributes.

-n, -N, -e, -E options can be used for attributes of a specific node, all nodes,
a specific edge, or all edges (respectively).  The syntax looks like 
 
 ... -N shape=egg -e NodeC LeafD color=red ...
  
-g can be used to specify whole graph options.  Eg:

 $ cat parsetree.txt | ./tree-to-graphvis -g 'ratio=1.2;' | dot -Tpng -o parsetree.png

"""

import sys
import re

def dot_xlate( t ) :
    d = { 
            'emptyset':'&empty;',
            'bullet':'&bull;',
            'lambda':'&lambda;',
            '->':'&rarr;',
            '<-':'&larr;',
            '=>':'&rArr;',
            '<=':'&lArr;',
            'times':'&times;',
            'otimes':'&otimes;',
            'oplus':'&oplus;',
            'spade':'&spades;',
            'spadesuit':'&spades;',
            'heart':'&#x2661;',      # &hearts; is a filled shape, hearts are red in a deck though...
            'heartsuit':'&#x2661;',
            'forall':'&forall;',
            'lowast':'&lowast;',
            'ast':'&lowast;',
            'diamond':'&diams;',
            'diamondsuit':'&diams;',
            'clubsuit':'&clubs;',
            'club':'&clubs;',
            'alpha':'&alpha;',
            'beta':'&beta;',
            'gamma':'&gamma;',
            # unicode
            r'\$':'&#36;',
            r'\\':'&#92;'
            }
    for k, i in d.items() :
        rc = re.compile( r'''(^|\s+|\\n|["',{])''' + k + r'''($|\s+|["',}\\])''' )
        t = rc.sub( r'\1'+i+r'\2', t)
    if t[0] == '"' and t[-1] == '"' :
        t = '\\' + t[:-1] + '\\' + t[-1]

    return t

def read( f, edgeprops ) :
    nodes = {}
    attrs = {}
    edges = []

    l = f.readline()
    while l :
        l = l.strip()
        if not l :
            l = f.readline()
            continue

        y = l.split( " . " )
        if len(y)==1 :
            nd, att = y[0], ""
        else :
            nd, att = y[0], " . ".join(y[1:]).strip()
        y = nd.split()
        if len(y)==1 :
            node, resid = y[0], [y[0],]
        else :
            node, resid = y[0], y[1:]

        if node in nodes :
            # edges specification
            assert( type(resid) == type([]) )
            for c in resid :
                edges.append( (node,c) )
                edgeprops.setdefault( (node,c), [] )
                if att :
                    if 'label=' in att :
                        edgeprops[ (node,c) ].append( 'labeldistance=3.0' )
                    edgeprops[ (node,c) ].append( att )
        else :
            # node definition and resid is description
            if type(resid) == type([]) :
                resid = " ".join( resid )
            nodes[node] = dot_xlate(resid)
            if att :
                attrs.setdefault(node,[]).append( dot_xlate( att ) )

        l = f.readline()

    return nodes, edges, edgeprops, attrs

# a little confusing, attrs for nodes come from the original input file,
# nodeprops come from command line
def _writenodes( f, nodes, nodeattrs, globnode, nodeprops, sep ) :
    for n, l in nodes.items() :
        a = nodeattrs.get( n, [] )
        a.extend( globnode )
        a.extend( nodeprops.get( n, [] ) )
        if not [t for t in a if t.startswith('label=')] :
            a.insert(0,'label="%s"'%(l,))
        f.write( '"%s" [%s];' % ( n, ",".join(a) ) )
        f.write( sep )

def _writeedges( f, edges, globedge, edgeprops, sep, edgesym ) :
    for n, c in edges :
        attrs = []
        attrs.extend( globedge )
        attrs.extend( edgeprops.get( (n,c), [] ))
        if attrs :
            f.write( '"%s" %s "%s" [%s];' % ( n, edgesym, c, ",".join(attrs) ) )
        else :
            f.write( '"%s" %s "%s";' % ( n, edgesym, c ) )
        f.write( sep )

def treewrite( f, nodes, edges, attrs, graphprops=[], globnode=[], nodeprops={}, globedge=[],
        edgeprops={}, sep="\n  ", edgesym='--' ) :
    f.write( "graph Pt {" + sep + sep.join( ["ordering=out;",] + graphprops ) + sep )
    _writenodes( f, nodes, attrs, globnode, nodeprops, sep )
    _writeedges( f, edges, globedge, edgeprops, sep, edgesym )
    f.write( "}\n" );

def fawrite( f, nodes, edges, attrs, graphprops=[], globnode=[], nodeprops={}, globedge=[], 
        edgeprops={}, sep="\n  ", edgesym='->' ) :
    f.write( "digraph Fa {" + (sep if graphprops else '') + sep.join( graphprops ) + sep )
    _writenodes( f, nodes, attrs, globnode, nodeprops, sep )
    _writeedges( f, edges, globedge, edgeprops, sep, edgesym )
    f.write( "}\n" );

def sc( gp ) :
    return [x if x.endswith(';') else x + ';' for x in gp]

import os

n = os.path.basename(sys.argv[0]).split('-')[0]
if n in ( 'tree', 'parsetree' ) :
    write=treewrite
elif n == 'fa' :
    write=fawrite
else :
    n=treewrite

sctail = re.compile( r';\s*$' )
output=None
infile=None
filecount=0
graphprops=[]
edgeprops={}
nodeprops={}
globedge=[]
globnode=[]
a=1
while a < len(sys.argv) :
    if sys.argv[a] in ("-n","--nodeprop",) :
        nodeprops.setdefault( sys.argv[a+1], [] ).append( dot_xlate(sys.argv[a+2]) )
        a+=3
        continue
    if sys.argv[a] in ("-N","--global-node",) :
        globnode.append( dot_xlate(sys.argv[a+1]) )
        a+=2
        continue
    if sys.argv[a] in ("-e","--edgeprop",) :
        edgeprops.setdefault( (sys.argv[a+1],sys.argv[a+2]), [] ).append( dot_xlate(sys.argv[a+3]) )
        a+=4
        continue
    if sys.argv[a] in ("-E","--global-edge",) :
        globedge.append( dot_xlate(sys.argv[a+1]) )
        a+=2
        continue
    if sys.argv[a] in ("-g","--graphprop",) :
        graphprops.append( sctail.sub( '', sys.argv[a+1] ) )
        a+=2
        continue
    if sys.argv[a] in ("-o","--output",) :
        output=sys.argv[a+1]
        a+=2
        continue
    if sys.argv[a] == "-" :
        n, e, edgeprops, att = read( sys.stdin, edgeprpos )
        write( sys.stdout if output in (None,"-") else open(output,"w"), n, e, att, 
                graphprops=sc(graphprops), globnode=globnode, nodeprops=nodeprops,
                globedge=globedge, edgeprops=edgeprops )
        output=None
        a+=1
        filecount+=1
        continue
    if not output :
        if sys.argv[a].endswith( ".gv" ) :
            # wierd, but don't overwrite
            output = sys.argv[a]+".gv"
        elif sys.argv[a] != "-" :
            output = os.path.splitext(sys.argv[a])[0] + ".gv"
    n, e, edgeprops, att = read( open(sys.argv[a]), edgeprops )
    write( sys.stdout if output in (None,"-") else open(output,"w"), n, e, att, 
            graphprops=sc(graphprops), globnode=globnode, nodeprops=nodeprops,
            globedge=globedge, edgeprops=edgeprops )
    output = None
    a+=1
    filecount+=1
    continue

if not filecount :
    n, e, edgeprops, att = read( sys.stdin, edgeprops )
    write( sys.stdout if output in (None,"-") else open(output,"w"), n, e, att, 
            graphprops=sc(graphprops), globnode=globnode, nodeprops=nodeprops,
            globedge=globedge, edgeprops=edgeprops )

sys.exit(0)

