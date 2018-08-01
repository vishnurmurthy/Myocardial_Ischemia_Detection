/* file: stclass.c

PhysioNet/Computers in Cardiology Challenge 2003

This file contains the main(), label(), open_files(), and close_input()
functions shared by all entries.  It should not be necessary to modify
this file.  Your implementations of the initialize(), analyze(), and
finalize() functions should be contained within a file named "analyze.c",
which will be included in this file when the classifier is compiled.

This program should compile without errors or warnings using:
        gcc -Wall stclass.c -lwfdb -lm
See http://www.physionet.org/physiotools/wfdb.shtml for information about
the WFDB library.

If your functions do not make use of the .dat, .atr, and .16a files, you
should be able to compile this program without the WFDB library, using:
        gcc -Wall -DNOWFDB stclass.c -lm

See http://www.physionet.org/challenge/2003/ for further information on
the PhysioNet/Computers in Cardiology Challenge 2003.
*/

#include <stdio.h>
#include <math.h>
#include <stdlib.h>
#include <string.h>
#include "analyze.c"	/* contains initialize(), analyze(), and finalize() */

#define MAX_ID	500	/* largest ST event ID number */
FILE *epifile, *epofile;
struct {	/* Data from epifile, to be written eventually to epofile */
    long time;		/* time of ST change, in seconds */
    int id;		/* ST change id (1, 2, ...) */
    int sig;		/* signal number (0, 1, or 2) */
    char label;		/* initially '?', label() changes to 'I' or 'N' */
} epi[MAX_ID+1];

void label(int event_id, char event_label)
{
    if (0 < event_id && event_id < MAX_ID)
	epi[event_id].label = event_label;
}

int main(int argc, char **argv)
{
    char buf[80];
    int epid, epsig, i;
    long eptime;
    void open_files(char *record);
    void close_input(void);

    /* The (six-character) record name must be the only argument on the
       command line. */
    if (argc != 2 || strlen(argv[1]) != 6) {
	fprintf(stderr, "usage: %s RECORD\n", argv[0]);
	exit(1);
    }

    /* Invoke the participant's initialize() function. */
    initialize();

    /* Open the input files and the .epo output file (argv[1] is the record
       name). */
    open_files(argv[1]);

    /* Load contents of the .epi file into memory. */
    while (fgets(buf, sizeof(buf), epifile)) {
	if (sscanf(buf, "%d %ld %d", &epid, &eptime, &epsig) == 3 &&
	    epid < MAX_ID) {
	    epi[epid].id = epid;
	    epi[epid].time = eptime;
	    epi[epid].sig = epsig;
	    epi[epid].label = '?';
	}
    }
    fclose(epifile);

    /* Invoke the participant's analyze() function once per event. */
    for (i = 1; i < MAX_ID; i++)
	if (epi[i].id == i)
	    analyze(i, epi[i].time, epi[i].sig);

    /* Close the input files. */
    close_input();

    /* Invoke the participant's finalize() function. */
    finalize();

    /* Write the .epo file. */
    for (i = 1; i < MAX_ID; i++)
	if (epi[i].id == i)
	    fprintf(epofile, "%3d %5ld %d %c\n", i, epi[i].time, epi[i].sig,
		    epi[i].label);
    fclose(epofile);

    exit(0);
}

void open_files(char *record)
{
    char epiname[16], eponame[16], kltname[16], stfname[16];

#if defined(USE_ANN)
    WFDB_Anninfo ai[2];
#endif

#if defined(USE_DAT) 
    WFDB_Siginfo si[3];

    nsig = isigopen(record, si, 3); /* LTSTDB records have 2 or 3 signals. */
    if (nsig < 2) exit(2);	     /* Quit if signals were not readable. */
#endif

#if defined(USE_ANN)
    ai[0].name = "atr"; ai[0].stat = WFDB_READ;
    ai[1].name = "16a"; ai[1].stat = WFDB_READ;
    if (annopen(record, ai, 2) < 0)
	exit(2);		     /* Quit if annotations are unreadable. */
#endif

#if defined(USE_STF)
    sprintf(stfname, "%s.stf", record);
    stffile = fopen(stfname, "rt");
    if (stffile == NULL) exit(2);     /* Quit if .stf file is unreadable. */
#endif

#if defined(USE_KLT)
    sprintf(kltname, "%s.klt", record);
    kltfile = fopen(kltname, "rt");
    if (kltfile == NULL) exit(2);     /* Quit if .klt file is unreadable. */
#endif

    sprintf(epiname, "%s.epi", record);
    epifile = fopen(epiname, "rt");
    if (epifile == NULL) exit(2);     /* Quit if .epi file is unreadable. */

    sprintf(eponame, "%s.epo", record);
    epofile = fopen(eponame, "wt");
    if (epofile == NULL) exit(2);     /* Quit if .epo file is unwritable. */

    return;
}

void close_input(void)
{
#if defined(USE_KLT)
    fclose(kltfile);
#endif

#if defined(USE_STF)
    fclose(stffile);
#endif

#if defined(USE_ANN) || defined(USE_DAT)
    wfdbquit();
#endif
}