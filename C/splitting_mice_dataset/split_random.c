#include <stdlib.h>
#include <stdio.h>

int main(int argc, char **argv)
{
    char *infilename = argv[1];
    printf("%s\n",infilename);

    FILE *ifp = fopen(infilename, "r");
    FILE *ofp = fopen("out.dat", "w");
    float ra,dec,z;

    if (ifp == NULL) {
        printf("Can't open input file %s!",infilename);
        exit(1);
    }


    if (ofp == NULL) {
        fprintf(stderr, "Can't open output file %s!\n", "out.dat");
        exit(1);
    }


    int count = 0;
    while (fscanf(ifp, "%f,%f,%f",&ra,&dec,&z) != EOF) {
          //printf("%f %f %f\n",ra,dec,z);
          fprintf(ofp,"%f %f %f\n",ra,dec,z);

          if (count%100000==0)
              printf("%d\n",count);

          count += 1;
    }

    fclose(ifp);
    fclose(ofp);

    return 0;
}
