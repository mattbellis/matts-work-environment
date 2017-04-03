#include<stdio.h>
#include<stdlib.h>

void display(int **board, int ncols, int nrows) {

    for(int j=0;j<nrows;j++) {
        for(int i=0;i<ncols;i++) {
            printf("%d ",board[i][j]);
        }
        printf("\n");
    }
}

int main() {

    int ncols = 7;
    int nrows = 6;
    int **board;

    // Allocate the memory for the blank board and 
    // initialize all the entries to be 0
    board  = (int **)malloc(sizeof(int *) * ncols);

    for(int i=0;i<ncols;i++) {
        board[i] = (int *)malloc(sizeof(int) * nrows);
        for(int j=0;j<nrows;j++) {
            board[i][j] = 0;
        }
    }

    display(board,ncols,nrows);


    // Input values
    int choice = 1;

    while(choice >= 1 && choice <= 7) {
        printf("Please select a column to drop your disc (1-7, anything else to exit)\n");
        scanf("%d", &choice);
        printf("You entered: %d\n", choice);

        for(int j=0;j<nrows;j++) {
            if(j==0 && board[choice-1][j] != 0) {
                printf("Column is filled! Pick another column.\n\n");
            }

            if(j<nrows-1 && board[choice-1][j+1] != 0) {
               board[choice-1][j] = 1;
               break;
            }

            else if(j==nrows-1 && board[choice-1][j] == 0) {
               board[choice-1][j] = 1;
            }
        }
        
        display(board,ncols,nrows);

    }

    //free display;

    return 0;
}
