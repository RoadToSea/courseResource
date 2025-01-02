#include <stdio.h>

int main() {
	int e[10][10], dis[10], book[10], n, m, t1, t2, t3, u, v, min;
	int inf = 99999999;
	scanf("%d %d", &n, &m);
	for (int i = 1; i <= n; i++) {
		for (int j = 1; j <= n; j++) {
			if (i == j)
				e[i][j] = 0;
			else
				e[i][j] = inf;
		}
	}
	//�����
	for (int i = 1; i <= m; i++) {
		scanf("%d %d %d", &t1, &t2, &t3);
		e[t1][t2] = t3;
	}
	//��ʼdis���ݣ�������1�Ŷ��㵽������������·��
	for (int i = 1; i <= n; i++) {
		dis[i] = e[1][i];
	}
	//book�����ʼ��
	for (int i = 1; i <= n; i++) {
		book[i] = 0;
	}
	book[1] = 1;
	//Dijkstra�㷨�������
	for (int i = 1; i <= n - 1; i++) {
		//�ҵ���һ�Ŷ�������Ķ���
		min = inf;
		for (int j = 1; j <= n; j++) {
			if (book[j] == 0 && dis[j] < min) {
				min = dis[j];
				u = j;
			}
		}
		book[u] = 1;
		for (v = 1; v <= n; v++) {
			if (e[u][v] < inf) {
				if (dis[v] > dis[u] + e[u][v]) {
					dis[v] = dis[u] + e[u][v];
				}
			}
		}
	}
	//������ս��
	for (int i = 1; i <= n; i++) {
		printf("����1��%d����̾���Ϊ��%5d\n", i, dis[i]);
	}
	getchar();
	return 0;
}
/*
6 9
1 2 1
1 3 12
2 3 9
2 4 3
3 5 5
4 3 4
4 5 13
4 6 15
5 6 4
*/
