#include <stdio.h>
#include <string.h>
#include <stdlib.h>

#define N 100
#define M 200//10*1000*1000

#define CO(x, y) ((y) * 2 + (x))

int find_first(int s, int* l, int i)
{
    for (int j = 0; j < N; ++j) {
	if (l[CO(i,j)] == s) {
	    return j;
	}
    }

    return -1;
}

void p23II(const char* input)
{
    int input_length = strlen(input);

    int* l = malloc(sizeof(int) * N * 2);

    for (int i = 0; i < N; ++i) {
	if (i < input_length) {
	    l[CO(0, i)] = input[i] - '0';
	}
	else {
	    l[CO(0, i)] = i + 1;
	}
    }

    int m = 1;
    int x = N;
    int n = x;
    int i = 0, j = 0;
    int pu[3] = {0, 0, 0};

    for (int move = 0; move < M; ++move)    
    {
	/* if (move <= 4) { */
	/*     for (int z = 0; z < N; ++z) { */
	/* 	printf("%d ", a[i][z]); */
	/*     } */
	/*     printf("\n"); */
	/* } */
	
	/* if (move % 1000 == 0) { */
	/*     printf("on move %d\n", move); */
	/* } */
	
	int s = l[CO(i,j)] - 1;
	if (s < m) s = x;
	int oi = (i+1) % 2;
	pu[0] = (j+1) % n, pu[1] = (j+2) % n, pu[2] = (j+3) % n;

	int k = -1;
	while (1) {
	    k = find_first(s, l, i);

	    if (k < 0 || k == pu[0] || k == pu[1] || k == pu[2]) {
		s = (s-1) < m ? x : (s-1);
		continue;
	    }
	    break;
	}

	int jj = j;
	l[CO(oi, jj)] = l[CO(i, jj)];
	int layed = 1;

	int jjo = (jj + 1) % n;
	int jji = (jj + 4) % n;
	while (1) {
	    l[CO(oi, jjo)] = l[CO(i, jji)];
	    ++layed;
	    if (jji == k) {
		jji = (jji + 1) % n, jjo = (jjo + 1) % n;
		break;
	    }
	    jji = (jji + 1) %n, jjo = (jjo + 1) % n;
	}

	for (int u = 0; u < 3; ++u) {
	    l[CO(oi, jjo)] = l[CO(i, pu[u])];
	    ++layed;
	    jjo = (jjo + 1) % n;
	}

	while (layed < n) {
	    l[CO(oi, jjo)] = l[CO(i, jji)];
            ++layed;
            jji = (jji + 1) % n, jjo = (jjo + 1) % n;
	}

        i = (i+1) % 2;
        j = (j+1) % n;
    }

    int j1 = find_first(1, l, i);
    int fl1 = l[CO(i, (j1+1) % n)];
    int fl2 = l[CO(i, (j1+2) % n)];
    long fl3 = fl1 * fl2;
    printf("%d * %d = %ld\n", fl1, fl2, fl3);
}

int main(int argc, char *argv[])
{
    p23II(argv[1]);
    return 0;
}
