% THESIS DOCUMENT-STYLE OPTION - Version 1.05, released 28 June 1995
%    for LaTeX version 2.09
% Copyright (C) 1994 by Marco Lusini (some parts (C) by C.Beccari)
%
% 03-12-94 (1.00) Prima release
% 17-12-94 (1.01) Modificato '\accademicyear'
% 10-01-95 (1.02) Modificato '\thebibliography', aggiunto '\hugeword' e
%                 '\preface'
% 07-06-95 (1.03) Aggiunto '\framepage' dal 'medaglione' del Beccari.
% 10-06-95 (1.04) Aggiunto '\listing', modificato dal Beccari.
% 28-06-95 (1.05) Modificato '@\makecaption' per centrare davvero
%                 le didascalie su una sola riga.
% 24-01-96 (1.06) Aggiunto '\introduction' per l'introduzione. Incluso il
%                 file report.cls. Modificato per il LaTeX 2e
%  9-02-96        Eliminato il correlatore (ad esclusivo uso della mia tesi)
%
% Questo stile modifica lo stile report nel seguente modo:
%  - le figure e le tabelle vengono preferibilmente messe dove
%    sono dichiarate (di default la posizione e' [htb])
%  - le didascalie sono composte in margini piu' stretti e con
%    un carattere piu' piccolo
%  - le testate sono composte dal nome del capitolo in piccolo,
%    allineate a destra e sottolineate
%  - il numero di pagina e' in basso al centro
%  - le prime pagine sono numerate in cifre romane e senza testate,
%    fino al primo capitolo (vedi anche l'ambiente 'romanpages')
%  - la bibliografia e' aggiunta all' indice.
%  - il comando '\maketitle' crea un frontespizio adatto ad una tesi,
%    usando le informazioni date dall' utente con i comandi:
%      \university    - Citta' in cui ha sede l'universita'
%      \faculty       - Denominazione della facolta'
%      \dept          - Nome del dipartimento
%      \course        - Denominazione del corso di laurea
%      \accademicyear - Anno accademico
%      \supervisor    - Relatore/i
%      \advisor       - Corelatore/i
%      \author        - Autore
%      \title         - Titolo
%    se non si usa \accademicyear, viene usato l'anno corrente
%    si puo' dichiarare piu' (co)relatori semplicemente ripetendo
%    il comando con i diversi nomi
%    Esempio:
%      \university{Firenze}
%      \faculty{Ingegneria}
%      \dept{Sistemi e Informatica}
%      \course{Ingegneria Elettronica}
%      \author{Marco Lusini}
%      \title{La supercazzora brematurata\\nella progettazione del software}
%      \supervisor{Pinco}
%      \supervisor{Pallino}
%      \advisor{Tizio}
%      \advisor{Caio}
%      \advisor{Sempronio}
%
% Inoltre sono state aggiunte le seguenti macro:
%  - comando di sezionamento '\preface', per scrivere la Prefazione ed
%    aggiungerla all' indice
%  - comando di sezionamento '\introduction', per scrivere l'Introduzione ed
%    aggiungerla all' indice
%  - ambiente 'romanpages' per definire in maniera alternativa al controllo
%    sul primo capitolo le prime pagine della tesi, quelle numerate in
%    numeri romani
%  - ambienti 'sidefigure', 'localfigure', 'sidetable', 'localtable',
%    per comporre figure a margine e fissate, compatibli con gli ambienti
%    'figure' e 'table':
%    - le versioni 'local' producono oggetti che non sono piu' flottanti,
%      ma che vengono collocati nel testo esattamente dove sono dichiarati
%    - le versioni 'side' producono oggetti allineati con il margine destro
%      della pagina, esattamente all'inizio del capoverso che le contiene e
%      il cui testo deve iniziare subito dopo il comando '\end{side..}'
%      senza che vi sia un fine capoverso esplicito (\par) o implicito.
%      Il capoverso che contiene l' oggetto laterale deve essere lungo
%      abbastanza da girargli intorno; per essere sicuri e' opportuno
%      iniziare un eventuale capoverso successivo con il comando '\sidepar'
%      che simula solamente l'inizio di un nuovo capoverso.
%  - '\hugeinitial', per comporre capolettera grandi
%  - '\hugeword', per comporre capolettera con la parola seguente in maiuscolo
%  - ambienti 'ftabular' e 'Ftabular', che non funzionano :-)
%  - ambiente 'framepage', crea una 'minipage' riquadrata
%  - comando '\listing' per includere listati 'verbatim'
%  - stile delle testate 'customheadings', in cui le testate
%    possono essere ridefinite (con molta attenzione) dall' utente
%
%
% Questo � il file report.cls originale (tranne dove indicato)

\NeedsTeXFormat{LaTeX2e}
\ProvidesClass{thesis}
              [1996/01/07 v1.06
 Non-standard LaTeX document class]
\newcommand\@ptsize{}
\newif\if@restonecol
\newif\if@titlepage
\@titlepagetrue
\newif\if@openright
\newif\if@openbib
\@openbibfalse
\DeclareOption{a4paper}
   {\setlength\paperheight {297mm}%
    \setlength\paperwidth  {210mm}}
\DeclareOption{a5paper}
   {\setlength\paperheight {210mm}%
    \setlength\paperwidth  {148mm}}
\DeclareOption{b5paper}
   {\setlength\paperheight {250mm}%
    \setlength\paperwidth  {176mm}}
\DeclareOption{letterpaper}
   {\setlength\paperheight {11in}%
    \setlength\paperwidth  {8.5in}}
\DeclareOption{legalpaper}
   {\setlength\paperheight {14in}%
    \setlength\paperwidth  {8.5in}}
\DeclareOption{executivepaper}
   {\setlength\paperheight {10.5in}%
    \setlength\paperwidth  {7.25in}}
\DeclareOption{landscape}
   {\setlength\@tempdima   {\paperheight}%
    \setlength\paperheight {\paperwidth}%
    \setlength\paperwidth  {\@tempdima}}
\DeclareOption{10pt}{\renewcommand\@ptsize{0}}
\DeclareOption{11pt}{\renewcommand\@ptsize{1}}
\DeclareOption{12pt}{\renewcommand\@ptsize{2}}
\DeclareOption{oneside}{\@twosidefalse \@mparswitchfalse}
\DeclareOption{twoside}{\@twosidetrue  \@mparswitchtrue}
\DeclareOption{draft}{\setlength\overfullrule{5pt}}
\DeclareOption{final}{\setlength\overfullrule{0pt}}
\DeclareOption{titlepage}{\@titlepagetrue}
\DeclareOption{notitlepage}{\@titlepagefalse}
\DeclareOption{openright}{\@openrighttrue}
\DeclareOption{openany}{\@openrightfalse}
\DeclareOption{onecolumn}{\@twocolumnfalse}
\DeclareOption{twocolumn}{\@twocolumntrue}
\DeclareOption{leqno}{\input{leqno.clo}}
\DeclareOption{fleqn}{\input{fleqn.clo}}
\DeclareOption{openbib}{\@openbibtrue}
\ExecuteOptions{letterpaper,10pt,oneside,onecolumn,final,openany}
\ProcessOptions
\input{size1\@ptsize.clo}
\setlength\lineskip{1\p@}
\setlength\normallineskip{1\p@}
\renewcommand\baselinestretch{}
\setlength\parskip{0\p@ \@plus \p@}
\@lowpenalty   51
\@medpenalty  151
\@highpenalty 301
\setcounter{topnumber}{2}
\renewcommand\topfraction{.7}
\setcounter{bottomnumber}{1}
\renewcommand\bottomfraction{.3}
\setcounter{totalnumber}{3}
\renewcommand\textfraction{.2}
\renewcommand\floatpagefraction{.5}
\setcounter{dbltopnumber}{2}
\renewcommand\dbltopfraction{.7}
\renewcommand\dblfloatpagefraction{.5}
\if@twoside
    \def\ps@headings{%
      \let\@oddfoot\@empty\let\@evenfoot\@empty
      \def\@evenhead{\thepage\hfil\slshape\leftmark}%
      \def\@oddhead{{\slshape\rightmark}\hfil\thepage}%
      \let\@mkboth\markboth
      \def\chaptermark##1{%
        \markboth {\uppercase{\ifnum \c@secnumdepth >\m@ne
            \@chapapp\ \thechapter. \ \fi
            ##1}}{}}%
      \def\sectionmark##1{%
        \markright {\uppercase{\ifnum \c@secnumdepth >\z@
            \thesection. \ \fi
            ##1}}}}
\else
    \def\ps@headings{%
      \let\@oddfoot\@empty
      \def\@oddhead{{\slshape\rightmark}\hfil\thepage}%
      \let\@mkboth\markboth
      \def\chaptermark##1{%
        \markright {\uppercase{\ifnum \c@secnumdepth >\m@ne
            \@chapapp\ \thechapter. \ \fi
            ##1}}}}
\fi
\def\ps@myheadings{%
    \let\@oddfoot\@empty\let\@evenfoot\@empty
    \def\@evenhead{\thepage\hfil\slshape\leftmark}%
    \def\@oddhead{{\slshape\rightmark}\hfil\thepage}%
    \let\@mkboth\@gobbletwo
    \let\chaptermark\@gobble
    \let\sectionmark\@gobble
    }
\if@titlepage
  \newcommand\maketitle{\begin{titlepage}%
  \let\footnotesize\small
  \let\footnoterule\relax
  \null\vfil
  \vskip 60\p@
  \begin{center}%
    {\LARGE \@title \par}%
    \vskip 3em%
    {\large
     \lineskip .75em%
      \begin{tabular}[t]{c}%
        \@author
      \end{tabular}\par}%
      \vskip 1.5em%
    {\large \@date \par}%       % Set date in \large size.
  \end{center}\par
  \@thanks
  \vfil\null
  \end{titlepage}%
  \setcounter{footnote}{0}%
  \let\thanks\relax\let\maketitle\relax
  \gdef\@thanks{}\gdef\@author{}\gdef\@title{}}
\else
\newcommand\maketitle{\par
  \begingroup
    \renewcommand\thefootnote{\fnsymbol{footnote}}%
    \def\@makefnmark{\hbox to\z@{$\m@th^{\@thefnmark}$\hss}}%
    \long\def\@makefntext##1{\parindent 1em\noindent
            \hbox to1.8em{\hss$\m@th^{\@thefnmark}$}##1}%
    \if@twocolumn
      \ifnum \col@number=\@ne
        \@maketitle
      \else
        \twocolumn[\@maketitle]%
      \fi
    \else
      \newpage
      \global\@topnum\z@   % Prevents figures from going at top of page.
      \@maketitle
    \fi
    \thispagestyle{plain}\@thanks
  \endgroup
  \setcounter{footnote}{0}%
  \let\thanks\relax
  \let\maketitle\relax\let\@maketitle\relax
  \gdef\@thanks{}\gdef\@author{}\gdef\@title{}}
\def\@maketitle{%
  \newpage
  \null
  \vskip 2em%
  \begin{center}%
    {\LARGE \@title \par}%
    \vskip 1.5em%
    {\large
      \lineskip .5em%
      \begin{tabular}[t]{c}%
        \@author
      \end{tabular}\par}%
    \vskip 1em%
    {\large \@date}%
  \end{center}%
  \par
  \vskip 1.5em}
\fi
\newcommand\chaptermark[1]{}
\setcounter{secnumdepth}{2}
\newcounter {part}
\newcounter {chapter}
\newcounter {section}[chapter]
\newcounter {subsection}[section]
\newcounter {subsubsection}[subsection]
\newcounter {paragraph}[subsubsection]
\newcounter {subparagraph}[paragraph]
\renewcommand\thepart          {\Roman{part}}
\renewcommand\thechapter       {\arabic{chapter}}
\renewcommand\thesection       {\thechapter.\arabic{section}}
\renewcommand\thesubsection    {\thesection.\arabic{subsection}}
\renewcommand\thesubsubsection {\thesubsection .\arabic{subsubsection}}
\renewcommand\theparagraph     {\thesubsubsection.\arabic{paragraph}}
\renewcommand\thesubparagraph  {\theparagraph.\arabic{subparagraph}}
\newcommand\@chapapp{\chaptername}
\newcommand\part{\cleardoublepage
                 \thispagestyle{plain}%
                 \if@twocolumn
                     \onecolumn
                     \@tempswatrue
                   \else
                     \@tempswafalse
                 \fi
                 \hbox{}\vfil
                 \secdef\@part\@spart}

\def\@part[#1]#2{%
    \ifnum \c@secnumdepth >-2\relax
      \refstepcounter{part}%
      \addcontentsline{toc}{part}{\thepart\hspace{1em}#1}%
    \else
      \addcontentsline{toc}{part}{#1}%
    \fi
    \markboth{}{}
    {\centering
     \interlinepenalty \@M
     \reset@font
     \ifnum \c@secnumdepth >-2\relax
       \huge\bfseries \partname~\thepart
       \par
       \vskip 20\p@
     \fi
     \Huge \bfseries #2\par}%
    \@endpart}
\def\@spart#1{%
    {\centering
     \interlinepenalty \@M
     \reset@font
     \Huge \bfseries #1\par}%
    \@endpart}
\def\@endpart{\vfil\newpage
              \if@twoside
                \hbox{}%
                \thispagestyle{empty}%
                \newpage
              \fi
              \if@tempswa
                \twocolumn
              \fi}
\newcommand\chapter{\if@openright\cleardoublepage\else\clearpage\fi
                    \thispagestyle{plain}%
                    \global\@topnum\z@
                    \@afterindentfalse
                    \secdef\@chapter\@schapter}
\def\@chapter[#1]#2{\ifnum \c@secnumdepth >\m@ne
                         \refstepcounter{chapter}%
                         \typeout{\@chapapp\space\thechapter.}%
                         \addcontentsline{toc}{chapter}%
                                   {\protect\numberline{\thechapter}#1}%
                    \else
                      \addcontentsline{toc}{chapter}{#1}
                    \fi
                    \chaptermark{#1}%
                    \addtocontents{lof}{\protect\addvspace{10\p@}}%
                    \addtocontents{lot}{\protect\addvspace{10\p@}}%
                    \if@twocolumn
                      \@topnewpage[\@makechapterhead{#2}]%
                    \else
                      \@makechapterhead{#2}%
                      \@afterheading
                    \fi}
\def\@makechapterhead#1{%
  \vspace*{50\p@}%
  {\parindent \z@ \raggedright \reset@font
    \ifnum \c@secnumdepth >\m@ne
         \huge\bfseries \@chapapp{} \thechapter
         \par
         \vskip 20\p@
       \fi
    \Huge \bfseries #1\par
    \nobreak
    \vskip 40\p@
  }}
\def\@schapter#1{\if@twocolumn
                   \@topnewpage[\@makeschapterhead{#1}]%
                 \else
                   \@makeschapterhead{#1}%
                   \@afterheading
                 \fi}
\def\@makeschapterhead#1{%
  \vspace*{50\p@}%
  {\parindent \z@ \raggedright
    \reset@font
    \Huge \bfseries  #1\par
    \nobreak
    \vskip 40\p@
  }}
\newcommand\section{\@startsection {section}{1}{\z@}%
                                   {-3.5ex \@plus -1ex \@minus -.2ex}%
                                   {2.3ex \@plus.2ex}%
                                   {\reset@font\Large\bfseries}}
\newcommand\subsection{\@startsection{subsection}{2}{\z@}%
                                     {-3.25ex\@plus -1ex \@minus -.2ex}%
                                     {1.5ex \@plus .2ex}%
                                     {\reset@font\large\bfseries}}
\newcommand\subsubsection{\@startsection{subsubsection}{3}{\z@}%
                                     {-3.25ex\@plus -1ex \@minus -.2ex}%
                                     {1.5ex \@plus .2ex}%
                                     {\reset@font\normalsize\bfseries}}
\newcommand\paragraph{\@startsection{paragraph}{4}{\z@}%
                                    {3.25ex \@plus1ex \@minus.2ex}%
                                    {-1em}%
                                    {\reset@font\normalsize\bfseries}}
\newcommand\subparagraph{\@startsection{subparagraph}{5}{\parindent}%
                                       {3.25ex \@plus1ex \@minus .2ex}%
                                       {-1em}%
                                      {\reset@font\normalsize\bfseries}}
\if@twocolumn
  \setlength\leftmargini  {2em}
\else
  \setlength\leftmargini  {2.5em}
\fi
\setlength\leftmarginii  {2.2em}
\setlength\leftmarginiii {1.87em}
\setlength\leftmarginiv  {1.7em}
\if@twocolumn
  \setlength\leftmarginv  {.5em}
  \setlength\leftmarginvi {.5em}
\else
  \setlength\leftmarginv  {1em}
  \setlength\leftmarginvi {1em}
\fi
\setlength\leftmargin    {\leftmargini}
\setlength  \labelsep  {.5em}
\setlength  \labelwidth{\leftmargini}
\addtolength\labelwidth{-\labelsep}
\@beginparpenalty -\@lowpenalty
\@endparpenalty   -\@lowpenalty
\@itempenalty     -\@lowpenalty
\renewcommand\theenumi{\arabic{enumi}}
\renewcommand\theenumii{\alph{enumii}}
\renewcommand\theenumiii{\roman{enumiii}}
\renewcommand\theenumiv{\Alph{enumiv}}
\newcommand\labelenumi{\theenumi.}
\newcommand\labelenumii{(\theenumii)}
\newcommand\labelenumiii{\theenumiii.}
\newcommand\labelenumiv{\theenumiv.}
\renewcommand\p@enumii{\theenumi}
\renewcommand\p@enumiii{\theenumi(\theenumii)}
\renewcommand\p@enumiv{\p@enumiii\theenumiii}
\newcommand\labelitemi{$\m@th\bullet$}
\newcommand\labelitemii{\normalfont\bfseries --}
\newcommand\labelitemiii{$\m@th\ast$}
\newcommand\labelitemiv{$\m@th\cdot$}
\newenvironment{description}
               {\list{}{\labelwidth\z@ \itemindent-\leftmargin
                        \let\makelabel\descriptionlabel}}
               {\endlist}
\newcommand\descriptionlabel[1]{\hspace\labelsep
                                \normalfont\bfseries #1}
\if@titlepage
  \newenvironment{abstract}{%
      \titlepage
      \null\vfil
      \begin{center}
        \bfseries \abstractname
      \end{center}}
     {\par\vfil\null\endtitlepage}
\else
  \newenvironment{abstract}{%
      \if@twocolumn
        \section*{\abstractname}%
      \else
        \small
        \begin{center}%
          {\bfseries \abstractname\vspace{-.5em}\vspace{\z@}}%
        \end{center}%
        \quotation
      \fi}
      {\if@twocolumn\else\endquotation\fi}
\fi
\newenvironment{verse}
               {\let\\=\@centercr
                \list{}{\itemsep      \z@
                        \itemindent   -1.5em%
                        \listparindent\itemindent
                        \rightmargin  \leftmargin
                        \advance\leftmargin 1.5em}%
                \item[]}
               {\endlist}
\newenvironment{quotation}
               {\list{}{\listparindent 1.5em%
                        \itemindent    \listparindent
                        \rightmargin   \leftmargin
                        \parsep        \z@ \@plus\p@}%
                \item[]}
               {\endlist}
\newenvironment{quote}
               {\list{}{\rightmargin\leftmargin}%
                \item[]}
               {\endlist}
\newenvironment{titlepage}
    {%
      \if@twocolumn
        \@restonecoltrue\onecolumn
      \else
        \@restonecolfalse\newpage
      \fi
      \thispagestyle{empty}%
      \if@compatibility
        \setcounter{page}{0}
      \else
        \setcounter{page}{1}%
      \fi}%
    {\if@restonecol\twocolumn \else \newpage \fi
      \setcounter{page}{1}%
    }
\newcommand\appendix{\par
  \setcounter{chapter}{0}%
  \setcounter{section}{0}%
  \renewcommand\@chapapp{\appendixname}%
  \renewcommand\thechapter{\Alph{chapter}}}
\setlength\arraycolsep{5\p@}
\setlength\tabcolsep{6\p@}
\setlength\arrayrulewidth{.4\p@}
\setlength\doublerulesep{2\p@}
\setlength\tabbingsep{\labelsep}
\skip\@mpfootins = \skip\footins
\setlength\fboxsep{3\p@}
\setlength\fboxrule{.4\p@}
\@addtoreset{equation}{chapter}
\renewcommand\theequation{\thechapter.\arabic{equation}}
\newcounter{figure}[chapter]
\renewcommand\thefigure{\thechapter.\@arabic\c@figure}
\def\fps@figure{tbp}
\def\ftype@figure{1}
\def\ext@figure{lof}
\def\fnum@figure{\figurename~\thefigure}
\newenvironment{figure}
               {\@float{figure}}
               {\end@float}
\newenvironment{figure*}
               {\@dblfloat{figure}}
               {\end@dblfloat}
\newcounter{table}[chapter]
\renewcommand\thetable{\thechapter.\@arabic\c@table}
\def\fps@table{tbp}
\def\ftype@table{2}
\def\ext@table{lot}
\def\fnum@table{\tablename~\thetable}
\newenvironment{table}
               {\@float{table}}
               {\end@float}
\newenvironment{table*}
               {\@dblfloat{table}}
               {\end@dblfloat}
\newlength\abovecaptionskip
\newlength\belowcaptionskip
\setlength\abovecaptionskip{10\p@}
\setlength\belowcaptionskip{0\p@}
\long\def\@makecaption#1#2{%
  \vskip\abovecaptionskip
  \sbox\@tempboxa{#1: #2}%
  \ifdim \wd\@tempboxa >\hsize
    #1: #2\par
  \else
    \hbox to\hsize{\hfil\box\@tempboxa\hfil}%
  \fi
  \vskip\belowcaptionskip}
\DeclareOldFontCommand{\rm}{\normalfont\rmfamily}{\mathrm}
\DeclareOldFontCommand{\sf}{\normalfont\sffamily}{\mathsf}
\DeclareOldFontCommand{\tt}{\normalfont\ttfamily}{\mathtt}
\DeclareOldFontCommand{\bf}{\normalfont\bfseries}{\mathbf}
\DeclareOldFontCommand{\it}{\normalfont\itshape}{\mathit}
\DeclareOldFontCommand{\sl}{\normalfont\slshape}{\@nomath\sl}
\DeclareOldFontCommand{\sc}{\normalfont\scshape}{\@nomath\sc}
\newcommand{\cal}{\protect\pcal}
\newcommand{\pcal}{\@fontswitch{\relax}{\mathcal}}
\newcommand{\mit}{\protect\pmit}
\newcommand{\pmit}{\@fontswitch{\relax}{\mathnormal}}
\newcommand\@pnumwidth{1.55em}
\newcommand\@tocrmarg {2.55em}
\newcommand\@dotsep{4.5}
\setcounter{tocdepth}{2}
\newcommand\tableofcontents{%
    \if@twocolumn
      \@restonecoltrue\onecolumn
    \else
      \@restonecolfalse
    \fi
    \chapter*{\contentsname
        \@mkboth{\uppercase{\contentsname}}{\uppercase{\contentsname}}}%
    \@starttoc{toc}%
    \if@restonecol\twocolumn\fi
    }
\newcommand\l@part[2]{%
  \ifnum \c@tocdepth >-2\relax
    \addpenalty{-\@highpenalty}%
    \addvspace{2.25em \@plus\p@}%
    \begingroup
      \setlength\@tempdima{3em}%
      \parindent \z@ \rightskip \@pnumwidth
      \parfillskip -\@pnumwidth
      {\leavevmode
       \large \bfseries #1\hfil \hbox to\@pnumwidth{\hss #2}}\par
       \nobreak
         \global\@nobreaktrue
         \everypar{\global\@nobreakfalse\everypar{}}
    \endgroup
  \fi}
\newcommand\l@chapter[2]{%
  \ifnum \c@tocdepth >\m@ne
    \addpenalty{-\@highpenalty}%
    \vskip 1.0em \@plus\p@
    \setlength\@tempdima{1.5em}%
    \begingroup
      \parindent \z@ \rightskip \@pnumwidth
      \parfillskip -\@pnumwidth
      \leavevmode \bfseries
      \advance\leftskip\@tempdima
      \hskip -\leftskip
      #1\nobreak\hfil \nobreak\hbox to\@pnumwidth{\hss #2}\par
      \penalty\@highpenalty
    \endgroup
  \fi}
\newcommand\l@section      {\@dottedtocline{1}{1.5em}{2.3em}}
\newcommand\l@subsection   {\@dottedtocline{2}{3.8em}{3.2em}}
\newcommand\l@subsubsection{\@dottedtocline{3}{7.0em}{4.1em}}
\newcommand\l@paragraph    {\@dottedtocline{4}{10em}{5em}}
\newcommand\l@subparagraph {\@dottedtocline{5}{12em}{6em}}
\newcommand\listoffigures{%
    \if@twocolumn
      \@restonecoltrue\onecolumn
    \else
      \@restonecolfalse
    \fi
    \chapter*{\listfigurename
      \@mkboth{\uppercase{\listfigurename}}%
              {\uppercase{\listfigurename}}}%
    \@starttoc{lof}%
    \if@restonecol\twocolumn\fi
    }
\newcommand\l@figure{\@dottedtocline{1}{1.5em}{2.3em}}
\newcommand\listoftables{%
    \if@twocolumn
      \@restonecoltrue\onecolumn
    \else
      \@restonecolfalse
    \fi
    \chapter*{\listtablename
      \@mkboth{\uppercase{\listtablename}}{\uppercase{\listtablename}}}%
    \@starttoc{lot}%
    \if@restonecol\twocolumn\fi
    }
\let\l@table\l@figure
\newdimen\bibindent
\bibindent=1.5em
\newenvironment{thebibliography}[1]
     {\chapter*{\bibname
        \@mkboth{\uppercase{\bibname}}{\uppercase{\bibname}}}%
      \list{\@biblabel{\arabic{enumiv}}}%
           {\settowidth\labelwidth{\@biblabel{#1}}%
            \leftmargin\labelwidth
            \advance\leftmargin\labelsep
            \if@openbib
              \advance\leftmargin\bibindent
              \itemindent -\bibindent
              \listparindent \itemindent
              \parsep \z@
            \fi
            \usecounter{enumiv}%
            \let\p@enumiv\@empty
            \renewcommand\theenumiv{\arabic{enumiv}}}%
      \if@openbib
        \renewcommand\newblock{\par}
      \else
        \renewcommand\newblock{\hskip .11em \@plus.33em \@minus.07em}%
      \fi
      \sloppy\clubpenalty4000\widowpenalty4000%
      \sfcode`\.=\@m}
     {\def\@noitemerr
       {\@latex@warning{Empty `thebibliography' environment}}%
      \endlist}
\newcommand\newblock{}
\newenvironment{theindex}
               {\if@twocolumn
                  \@restonecolfalse
                \else
                  \@restonecoltrue
                \fi
                \columnseprule \z@
                \columnsep 35\p@
                \twocolumn[\@makeschapterhead{\indexname}]%
                \@mkboth{\uppercase{\indexname}}%
                        {\uppercase{\indexname}}%
                \thispagestyle{plain}\parindent\z@
                \parskip\z@ \@plus .3\p@\relax
                \let\item\@idxitem}
               {\if@restonecol\onecolumn\else\clearpage\fi}
\newcommand\@idxitem  {\par\hangindent 40\p@}
\newcommand\subitem   {\par\hangindent 40\p@ \hspace*{20\p@}}
\newcommand\subsubitem{\par\hangindent 40\p@ \hspace*{30\p@}}
\newcommand\indexspace{\par \vskip 10\p@ \@plus5\p@ \@minus3\p@\relax}
\renewcommand\footnoterule{%
  \kern-3\p@
  \hrule width .4\columnwidth
  \kern 2.6\p@}
\@addtoreset{footnote}{chapter}
\long\def\@makefntext#1{%
    \parindent 1em%
    \noindent
    \hbox to 1.8em{\hss\@makefnmark}#1}
\newcommand\contentsname{Contents}
\newcommand\listfigurename{List of Figures}
\newcommand\listtablename{List of Tables}
\newcommand\bibname{Bibliography}
\newcommand\indexname{Index}
\newcommand\figurename{Figure}
\newcommand\tablename{Table}
\newcommand\partname{Part}
\newcommand\chaptername{Chapter}
\newcommand\appendixname{Appendix}
\newcommand\abstractname{Abstract}
\newcommand\today{\ifcase\month\or
  January\or February\or March\or April\or May\or June\or
  July\or August\or September\or October\or November\or December\fi
  \space\number\day, \number\year}
\setlength\columnsep{10\p@}
\setlength\columnseprule{0\p@}
\pagestyle{plain}
\pagenumbering{arabic}      % Arabic page numbers
\if@twoside
\else
  \raggedbottom
\fi
\if@twocolumn
  \twocolumn
  \sloppy
  \flushbottom
\else
  \onecolumn
\fi
%\endinput
%%
%% End of file `report.cls'.

%
% Posizione di default per figure e tabelle
%
\def\fps@figure{htb}
\def\fps@table{thb}

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%
% Comando per l' inclusione di listati (Cfr. Beccari, 187-188)
%
\def\tabnumber{8}%                       Valore di default
%
\newdimen\tabwidth
\setbox0=\hbox{\footnotesize\tt\space}
\tabwidth=\tabnumber\wd0%                Valore di default
%
\def\listing#1{\pgm@lig \par\begingroup \programma \input #1 \endgroup \old@lig}
%
%
%
% Comando per rendere tutti i caratteri speciali dei cartteri ordinari
%
\def\uncatcodespecials{\def\do##1{\catcode`##1=12}\dospecials}
%
% \footnotesize puo' essere sostituito con \small o anche eliminato
%
\def\programma{%
   \setbox0=\hbox{\footnotesize\tt\space}
   \tabwidth=\tabnumber\wd0
   \def\baselinestretch{1}\footnotesize\tt \parindent 0pt
   \def\par{\leavevmode\egroup\box0\endgraf}
   \catcode`\`=\active% Gli apici non devono formare legature
   \catcode`\'=\active
   \catcode`\^^I=\active% <TAB> deve funzionare come <TAB>
   \catcode`\^^L=\active% <FF> deve eseguire un salto di pagina
   \obeylines \uncatcodespecials
   \obeyspaces \everypar{\startbox}
}
%
\def\startbox{\setbox0=\hbox\bgroup}
%
{\obeyspaces\global\let =\ }
%
% Si salvano le vecchie definizioni di ' e ` e si sostituiscono
% con delle definizioni assegnabili facilmente.
%
{\catcode`\`=\active \global\let\@oldlq=` \gdef`{\@mylq} }
{\catcode`\'=\active \global\let\@oldrq=' \gdef'{\@myrq} }
%
\gdef\@pgmlq{\relax\lq}
\gdef\@pgmrq{\relax\rq}
%
\def\pgm@lig{ \global\let\@mylq\@pgmlq \global\let\@myrq\@pgmrq }
\def\old@lig{ \global\let\@mylq\@oldlq \global\let\@myrq\@oldrq }
\old@lig
%
%  <TAB> agisce mediante la formazione di scatole {\box0} la
%       cui larghezza viene imposta ad un multiplo di \tabwidth
%
{\catcode`\^^I=\active \gdef^^I{\leavevmode\egroup
    \dimen0=\wd0
    \divide\dimen0 by\tabwidth
    \multiply\dimen0 by\tabwidth
    \advance\dimen0 by\tabwidth
    \wd0=\dimen0 \box0 \startbox}}
%
%   <FF> agisce terminando la riga ed inserendo una penalita'
%    negativa di valore tale che la routine di uscita di TeX
%    la interpreta come l'ordine di eseguire un salto pagina.
%                     (e' esattamente quello che fa \newpage)
%
{\catcode`\^^L=\active \gdef^^L{\par\penalty-10000}}
%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%
% Comando per la composizione della didascalia delle
% figure, delle tabelle e degli altri oggetti flottanti
%
% Si tratta della ridefinizione di un comando interno di LaTeX
% (Cfr. Beccari, 219)
%
\long\def\@makecaption#1#2{\begingroup%
   \vskip 10pt              % il LaTeX2e usa \abovecaptionskip invece di 10pt
   \linewidth \hsize
   \advance\linewidth -3em
   \def\baselinestretch{0.9}
   \small
   \setbox\@tempboxa\hbox{#2}%
   \ifdim \wd\@tempboxa=\z@
      \setbox\@tempboxa\hbox{#1}%
   \else
      \setbox\@tempboxa\hbox{#1. #2}%
   \fi
   \ifdim \wd\@tempboxa >\linewidth
      \advance\leftskip1.5em
      \advance\rightskip1.5em
      \unhbox\@tempboxa\par%
   \else
      \hbox to\hsize{\hfil\box\@tempboxa\hfil}%
   \fi                      % il LaTeX2e aggiunge \vskip\belowcaptionskip
\endgroup}
%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%
% Comandi per figure e tabelle fisse (cfr. Beccari, 227 e segg.):
%
% \begin{sidefigure}
% ...
% \end{sidefigure}
%
% \begin{localfigure}
% ...
% \end{localfigure}
%
%
\newdimen\sideftwd
\newbox\local@box\newbox\local@hbox
\def\sidepar{\newline\hspace*{\parindent}}
\def\sidefigure{\if@twocolumn\local@ft{figure}\else\side@ft{figure}\fi}
\def\endsidefigure{\endside@ft}
\def\sidetable{\if@twocolumn\local@ft{table}\else\side@ft{table}\fi}
\def\endsidetable{\endside@ft}
%
\def\side@ft#1{\par
\sideftwd=\textwidth \divide\sideftwd by 2
\def\@captype{#1}
\setbox\@tempboxa\vtop\bgroup\textwidth=\sideftwd
\columnwidth=\sideftwd \hsize\columnwidth
\@parboxrestore}
%
\def\endside@ft{\egroup
\@tempdima=\ht\@tempboxa
\advance\@tempdima by \dp\@tempboxa
\@tempcnta=\@tempdima
\divide\@tempcnta by \baselineskip
\advance\@tempcnta by 2
\global\hangindent-\sideftwd \global\hangafter-\@tempcnta
\noindent \dp\@tempboxa=\z@ \ht\@tempboxa=\z@
\hbox to \z@{\hskip\textwidth
\hbox to \z@{\hss\box\@tempboxa}\hss}%
\hskip\parindent
\global\@ignoretrue}
%
\def\local@ft#1{\def\@captype{#1}
\setbox\local@box\vbox\bgroup
\boxmaxdepth\z@ \hsize0.9\columnwidth}
\def\endlocal@ft{\egroup
\[\hbox{\lower1ex\box\local@box}\]
\global\@ignoretrue}
%
\def\localfigure{\local@ft{figure}}
\def\endlocalfigure{\endlocal@ft}
\def\localtable{\local@ft{table}}
\def\endlocaltable{\endlocal@ft}
%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%
% Ambiente per minipage riquadrate (Cfr. Beccari, 260)
%
% \begin{framepage}
% ...
% \end{framepage}
\newenvironment{framepage}[1]{% Comandi di apertura
\setbox0=\hbox\bgroup
\begin{minipage}{#1}\parindent=1em
}{%                             Comandi di chiusura
\end{minipage}\egroup
\fboxsep=0.5em\fbox{\box0}}
%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%
% \hugeinitial (Cfr. Beccari, 283-284)
% Modificato per allineare (o quasi) il capolettera alla prima riga.
% \hugeword compone la prima parola in maiuscolo, con la prima lettera grande.
%
\newcommand{\hugeinitial}[1]{%
\indent{\setbox0\lastbox}%
\dimen0\parindent
\setbox0\hbox{\Huge#1}%                 % era: {\huge#1}
\ifdim\wd0>\dimen0 \dimen0=\wd0\fi
\hangindent\dimen0
\dimen0=\ht0
%\advance\dimen0\dp0
%\advance\dimen0 by -1.58ex             % usare questa per allineare alle maiuscole
\advance\dimen0 by -\ht\strutbox        % era: \divide\dimen0 by 5
\setbox0\hbox to \z@{\hss\lower\dimen0\box0}%
\ifdim\dp0>0.3\baselineskip
\hangafter-2\else\hangafter-1\fi
\dp0=0pt\box0}
%
\def\hugeword#1#2 {\hugeinitial#1 \uppercase{#2} }
%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%
% ftabular e Ftabular (Cfr. Beccari, 261) (non funzionano!)
%
\newenvironment{ftabular}{%     Comandi di apertura
\setlength{\fboxsep}{\doublerulesep}
\fbox\bgroup\tabular%
}{%                             Comandi di chiusura
\endtabular\egroup
}
%
\newenvironment{Ftabular}{%     Comandi di apertura
\setlength{\fboxsep}{0pt}
\setlength{\fboxrule}{3\arrayrulewidth}
\fbox\bgroup\tabular%
}{%                             Comandi di chiusura
\endtabular\egroup}
%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%
% stile customheadings (cfr. Beccari, 311 e segg.)
%
\def\ps@customheadings{\let\@mkboth\markboth
\def\@evenhead{\makeevenhead}%  % Modifiche alle
\def\@oddhead{\makeoddhead}%    % definizioni
\def\@evenfoot{\makeevenfoot}%  % originali
\def\@oddfoot{\makeoddfoot}%    %
\def\chaptermark##1{\markboth
{\uppercase{\ifnum \c@secnumdepth>\m@ne
 \@chapapp\ \thechapter. \ \fi ##1}}{}}%
\def\sectionmark##1{\markright
{\uppercase{\ifnum \c@secnumdepth >\z@
 \thesection. \ \fi ##1}}}}
%
% Definizioni di default
%
\def\underlinehdr#1{\vbox{\hbox to \textwidth {#1\strut} \hbox to \textwidth {\hrulefill} \vskip -0.4pt}}
\def\makeevenhead{\underlinehdr{\def\baselinestretch{1}\footnotesize\sl \leftmark \hfil \null}}
\def\makeoddhead{\underlinehdr{\def\baselinestretch{1}\footnotesize\sl \null \hfil \leftmark}}
\def\makeevenfoot{}
\def\makeoddfoot{\null\hfil\rm\thepage\hfil\null}
%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%
% Modifica di \chapter per cambiare stile di pagina al primo capitolo,
% in modo che le prime pagine siano con i numeri romani e senza testate
% fino al primo capitolo, quando si passa ai numeri arabi.
%
\let\@riginalchapter\chapter
\def\chapter{%
\@ifstar{\@riginalchapter*}{%
\ifnum\thechapter=0%
\newpage\pagenumbering{arabic}\pagestyle{customheadings}%
\global\let\chapter\@riginalchapter%
\fi%
\@riginalchapter}%
}

\ps@plain
\pagenumbering{roman}

%
% Ambiente 'romanpages', usato in alternativa al meccanismo di cui sopra
% per definire le prime pagine della tesi con i numeri romani e niente
% testate.
%
\def\romanpages{\pagestyle{plain}\pagenumbering{roman}\global\let\chapter\@riginalchapter}
\def\endromanpages{\newpage\pagenumbering{arabic}\pagestyle{customheadings}%
\global\let\romanpages\relax\global\let\endromanpages\relax}
%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%
% Modifica di \thebibliography per aggiungere la bibliografia all' indice
%
\let\@riginalbib\thebibliography
\def\thebibliography#1{%
\@riginalbib{#1}%
\addcontentsline{toc}{chapter}{\bibname}%
}
%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%
% Comando '\preface', per comporre la prefazione
%
\def\preface{
  \chapter*{\prefacename
        \@mkboth{\uppercase{\prefacename}}{\uppercase{\prefacename}}}%
  \addcontentsline{toc}{chapter}{\prefacename}%
}
%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%
% Comando '\introduction', per comporre l'introduzione
%
\def\introductionname{Introduzione}
\def\introduction{
  \chapter*{\introductionname
        \@mkboth{\uppercase{\introductionname}}{\uppercase{\introductionname}}}%
  \addcontentsline{toc}{chapter}{\introductionname}%
}
%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%
% Comandi per la spaziatura doppia (da setspace.sty)
%
\def\doublespace{%
  \ifcase \@ptsize \relax % 10pt
    \def\baselinestretch{1.667}
  \or % 11pt
    \def\baselinestretch{1.618}
  \or % 12pt
    \def\baselinestretch{1.655}
  \fi
  \ifx\@currsize\normalsize\@normalsize\else\@currsize\fi%
}
%
\def\onehalfspace{%
  \ifcase \@ptsize \relax % 10pt
    \def\baselinestretch{1.25}
  \or % 11pt
    \def\baselinestretch{1.213}
  \or % 12pt
    \def\baselinestretch{1.241}
  \fi
  \ifx\@currsize\normalsize\@normalsize\else\@currsize\fi%
}
%
\def\singlespace{%
  \def\baselinestretch{1}
  \ifx\@currsize\normalsize\@normalsize\else\@currsize\fi%
%  \vskip\baselineskip% Correction for coming into singlespace
}

%\let\@riginalxfloat\@xfloat
%\def\@xfloat{%
%\singlespace%
%\@riginalxfloat%
%}
%\let\@riginalfootnotetext\@footnotetext
%\long\def\@footnotetext#1{%
%{\singlespace%
%\@riginalfootnotetext{#1}}%
%}
%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%
% Comandi per il nuovo maketitle, adatto alle tesi di laurea
%

% Parti dipendenti dalla lingua
\def\universityname{Universit\`a degli Studi di }
\def\facultyname{Scuola di }
\def\deptname{Dipartimento di }
\def\coursename{Tesi di laurea in }
\def\accademicyearname{Anno Accademico }
\def\authorname{Candidato}
\def\supervisorname{Relatore}
\def\supervisorsname{Relatori}
\def\advisorsname{Correlatori}
\def\advisorname{Correlatore}

%
% Definizione di default per l' A.A.: fino ad Aprile: Anno-2--Anno-1
%                                     dopo Aprile: Anno-1--Anno
\gdef\@accademicyear{
       \@tempcnta=\year%
       \advance\@tempcnta by -1
       \ifnum\month>4
          \number\the\@tempcnta --\number\the\year
       \else
          \@tempcntb=\year%
          \advance\@tempcntb by -2
          \number\the\@tempcntb --\number\the\@tempcnta
       \fi
}
\gdef\@university{}
\gdef\@faculty{}
\gdef\@dept{}
\gdef\@course{}
\gdef\@supervisor{}
\gdef\@advisor{}

% Comandi per i parametri del frontespizio
\def\university#1{\gdef\@university{#1}}
\def\faculty#1{\gdef\@faculty{#1}}
\def\dept#1{\gdef\@dept{#1}}
\def\course#1{\gdef\@course{#1}}
\def\accademicyear#1{\gdef\@accademicyear{#1}}
\def\supervisor#1{\ifx \empty\@supervisor
                 \gdef\@supervisor{#1}
               \else
                 \gdef\supervisorname{\supervisorsname}
                 \xdef\@supervisor{ \@supervisor \vskip 1.5ex {#1}}
               \fi}
\def\advisor#1{\ifx \empty\@advisor
                 \gdef\@advisor{#1}
               \else
                 \gdef\advisorname{\advisorsname}
                 \xdef\@advisor{ \@advisor \vskip 1.5ex {#1} }
               \fi}

%
% Ridefinizione del comando \maketitle per generare il frontespizio
%
\def\maketitle{%
\begin{titlepage}%
 \centerline{\includegraphics[scale=0.17]{imgs/stemma}}
 \setcounter{page}{0}%
 {\centering%    Intestazione centrata
%                Universita' (\large)
 \large\rm
 \uppercase\expandafter{\universityname}\uppercase\expandafter{\@university} \par%
%                Facolta' (\normalsize)
 \small\sc
 \facultyname \@faculty\ - \deptname \@dept \par
 \rule{7em}{0.4pt}
%                Corso di laurea (\normalsize)
 \vskip 1ex
 \normalsize\rm
 \coursename \@course \par
 }
%                Titolo centrato (\LARGE)
 \vskip 33mm
 {\centering \LARGE\sc \@title \par}%
 \vfill
%                Spazio per le firme (\normalsize)
 \vskip 20mm
 \vbox to 65mm {
 \normalsize\rm
 \hfil%
 \begin{minipage}[t]{65mm}% Autore sulla Sx
   \raggedright
   \it\authorname\\
   \rm\@author
 \end{minipage}%
%\hfil%
 \begin{minipage}[t]{50mm}% Relatori sulla Dx
   \raggedright
   \it\supervisorname\\
   \rm\@supervisor\\
   \vskip 3ex
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% Solo per la mia tesi!!
%
   \it\advisorname\\
   \rm\@advisor
%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
 \end{minipage}%
 \hfil\vfil}%
%                Anno accademico (\large)
 \vbox to 10mm {
 \centering
 \rule{7em}{0.4pt}
 \vskip 1.3ex
 \large\sc \accademicyearname \@accademicyear \par
 \vfil}%
\end{titlepage}%
\global\let\maketitle\relax
}
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
\endinput
