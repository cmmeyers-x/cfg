### CFG : Simple LL(1) Parser

# ABOUT
This project is a collaborative project for CSCI425: Compiler Design *TODO TODO*  

# USAGE
`python parser.py <grammar config> [token stream]`  

Grammar config defines a language in plain text (file extension: `.cfg`) EX:  
```
S -> A C $
C -> c
   | lambda
A -> a B C d
   | B Q
   | lambda
B -> b B | d
Q -> q
```
All grammar terminals are assumed to be lowercase)
`->` denotes the production rule associate
`|` is reserved for rule alternation
`lambda` specifies the empty string  

  
Language config files define a stream of language tokens (intermediate format void a lexer framework; extension: `.tok`)  
Each token is separated by newlines, each line containing either TOKEN or TOKEN TOKENVALUE  

### EXAMPLE
`python parser.py config/language-slides/language.cfg config/language-slides/language.tok`
