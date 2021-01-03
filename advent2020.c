#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include <assert.h>

#define N (1000*1000)
#define M (10*1000*1000)

struct LinkedList {
    struct LinkedList* nxt;
    int val;
};
typedef struct LinkedList Node;

Node* find_first(Node* ll, int s)
{
    Node* d = ll;
    for (int i = 0; i < N; ++i) {
	if (s == d->val) {
	    return d;
	}
	d = d->nxt;
    }

    assert(0);
    return NULL;
}

void dump_list(Node* ll)
{
    Node* cur = ll;
    for (int i = 0; i < N; ++i) {
	printf("%d ", cur->val);
	cur = cur->nxt;
    }
    printf("\n");
}

void p23II(const char* input)
{
    int input_length = strlen(input);

    Node* ll = malloc(sizeof(Node) * N);
    
    for (int i = 0; i < N; ++i) {
	if (i < input_length) {
	    ll[i].val = input[i] - '0';
	}
	else {
	    ll[i].val = i + 1;
	}

	ll[i].nxt = &ll[(i+1) % N];
    }

    Node* cur = &ll[0];
    for (int move = 0; move < M; ++move)    
    {
	/* printf("start: "); */
	/* dump_list(ll); */
	
	Node* nxt1 = cur->nxt;
	Node* nxt2 = nxt1->nxt;
	Node* nxt3 = nxt2->nxt;
	Node* nxt4 = nxt3->nxt;

	int s = cur->val;
	do {
	    --s;
	    if (s < 1) s = N;
	} while (s == nxt1->val || s == nxt2->val || s == nxt3->val);
	
	Node* dst = find_first(ll, s);
	assert(dst && dst != nxt1 && dst != nxt2 && dst != nxt3);

	cur->nxt = nxt4;

	Node* ndst = dst->nxt;
	dst->nxt = nxt1;
	nxt3->nxt = ndst;

	cur = nxt4;
	/* printf("stop: "); */
	/* dump_list(ll); */
    }

    Node* j1 = find_first(ll, 1);
    Node* j1nxt1 = j1->nxt;
    Node* j1nxt2 = j1nxt1->nxt;
    int fl1 = j1nxt1->val;
    int fl2 = j1nxt2->val;
    int fl3 = fl1 * fl2;
    printf("%d * %d = %ld\n", fl1, fl2, fl3);
}

int main(int argc, char *argv[])
{
    p23II(argv[1]);
    return 0;
}
