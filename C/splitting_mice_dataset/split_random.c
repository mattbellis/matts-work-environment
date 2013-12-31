#include <stdlib.h>
#include <stdio.h>

int main(int argc, char **argv)
{
    char *infilename = argv[1];
    printf("%s\n",infilename);

    char outfilename[256];
    sprintf(outfilename,"outfile_nentries%d_%04d.dat",nentries_per_file,filecount);

    FILE *ifp = fopen(infilename, "r");
    FILE *ofp = NULL;
    float ra,dec,z;

    if (ifp == NULL) {
        printf("Can't open input file %s!",infilename);
        exit(1);
    }

    printf("Opening %s\n",outfilename);
    ofp = fopen(outfilename, "w");
    if (ofp == NULL) {
        fprintf(stderr, "Can't open output file %s!\n", "out.dat");
        exit(1);
    }
    int count = 0;

    float probability_of_accepting_an_entry = 0.005;

    while (fscanf(ifp, "%f,%f,%f",&ra,&dec,&z) != EOF) {
        //printf("%f %f %f\n",ra,dec,z);
        if (count%nentries_per_file==0)
        {
            filecount += 1;
        }
        fprintf(ofp,"%f %f %f\n",ra,dec,z);

        if (count%100000==0)
            printf("%d\n",count);

        count += 1;
        if (count%nentries_per_file==nentries_per_file-1)
        {
            close(ofp);
        }
    }

    fclose(ifp);
    fclose(ofp);

    return 0;
}
