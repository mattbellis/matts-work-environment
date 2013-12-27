#include <stdlib.h>
#include <stdio.h>

int main(int argc, char **argv)
{
    char *infilename = argv[1];
    printf("%s\n",infilename);

    char outfilename[256];

    FILE *ifp = fopen(infilename, "r");
    FILE *ofp = NULL;
    float ra,dec,z;
    float z_v,x_c,y_c,z_c;
    char dummy[256];

    if (ifp == NULL) {
        printf("Can't open input file %s!",infilename);
        exit(1);
    }

    int count = 0;
    int filecount = 0;
    int nentries_per_file = 1000000;

    fscanf(ifp, "%s\t%s\t%s\t%s\t%s\t%s\t%s",&dummy,&dummy,&dummy,&dummy,&dummy,&dummy,&dummy);

    //while (fscanf(ifp, "%f,%f,%f",&ra,&dec,&z) != EOF) {
    while (fscanf(ifp, "%f\t%f\t%f\t%f\t%f\t%f\t%f",&ra,&dec,&z,&z_v,&x_c,&y_c,&z_c) != EOF) {
        //printf("%f %f %f\n",ra,dec,z);
        if (count%nentries_per_file==0)
        {
            sprintf(outfilename,"smaller_file_nentries%d_%04d.dat",nentries_per_file,filecount);
            printf("Opening %s\n",outfilename);
            ofp = fopen(outfilename, "w");
            if (ofp == NULL) {
                fprintf(stderr, "Can't open output file %s!\n", "out.dat");
                exit(1);
            }
            filecount += 1;
        }
        //fprintf(ofp,"%f %f %f\n",ra,dec,z);
        fprintf(ofp,"%f %f %f %f %f %f %f\n",ra,dec,z,z_v,x_c,y_c,z_c);

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
