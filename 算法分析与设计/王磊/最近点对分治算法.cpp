#include <stdio.h>
#include <stdlib.h>
#include <math.h>

// �����ṹ��
typedef struct {
	double x, y;
} Point;

// ���������Խṹ��
typedef struct {
	Point p1, p2;
	double distance;
} ClosestPair;

// �ȽϺ�������x��������
int compareX(const void *a, const void *b) {
	Point *p1 = (Point *)a;
	Point *p2 = (Point *)b;
	return (p1->x > p2->x) - (p1->x < p2->x);
}

// �ȽϺ�������y��������
int compareY(const void *a, const void *b) {
	Point *p1 = (Point *)a;
	Point *p2 = (Point *)b;
	return (p1->y > p2->y) - (p1->y < p2->y);
}

// ��������֮��ľ���
double dist(Point p1, Point p2) {
	return sqrt((p1.x - p2.x) * (p1.x - p2.x) + (p1.y - p2.y) * (p1.y - p2.y));
}

// ������������С����
ClosestPair bruteForce(Point P[], int n) {
	ClosestPair minPair;
	minPair.distance = __DBL_MAX__;
	for (int i = 0; i < n; ++i) {
		for (int j = i + 1; j < n; ++j) {
			double d = dist(P[i], P[j]);
			if (d < minPair.distance) {
				minPair.distance = d;
				minPair.p1 = P[i];
				minPair.p2 = P[j];
			}
		}
	}
	return minPair;
}

// �����������е���Сֵ
ClosestPair minPair(ClosestPair p1, ClosestPair p2) {
	return (p1.distance < p2.distance) ? p1 : p2;
}

// ����strip��������С����
ClosestPair stripClosest(Point strip[], int size, ClosestPair d) {
	qsort(strip, size, sizeof(Point), compareY);
	for (int i = 0; i < size; ++i) {
		for (int j = i + 1; j < size && (strip[j].y - strip[i].y) < d.distance; ++j) {
			double dist = sqrt((strip[i].x - strip[j].x) * (strip[i].x - strip[j].x) +
			                   (strip[i].y - strip[j].y) * (strip[i].y - strip[j].y));
			if (dist < d.distance) {
				d.distance = dist;
				d.p1 = strip[i];
				d.p2 = strip[j];
			}
		}
	}
	return d;
}

// ���η�������С����
ClosestPair closestUtil(Point P[], int n) {
	if (n <= 3)
		return bruteForce(P, n);

	int mid = n / 2;
	Point midPoint = P[mid];

	ClosestPair dl = closestUtil(P, mid);
	ClosestPair dr = closestUtil(P + mid, n - mid);

	ClosestPair d = minPair(dl, dr);

	Point strip[n];
	int j = 0;
	for (int i = 0; i < n; i++)
		if (fabs(P[i].x - midPoint.x) < d.distance)
			strip[j] = P[i], j++;

	return minPair(d, stripClosest(strip, j, d));
}

// ������������������
ClosestPair closest(Point P[], int n) {
	qsort(P, n, sizeof(Point), compareX);
	return closestUtil(P, n);
}

int main() {
	// ����1������һ��ֱ����
	Point P1[] = {{0.0, 0.0}, {1.0, 1.0}, {2.0, 2.0}, {3.0, 3.0}, {4.0, 4.0}, {5.0, 5.0}};
	int n1 = sizeof(P1) / sizeof(P1[0]);
	ClosestPair result1 = closest(P1, n1);
	printf("����1:\n");
	printf("��̾����� %f ���������ǣ� (%f, %f) (%f, %f)\n",
	       result1.distance, result1.p1.x, result1.p1.y, result1.p2.x, result1.p2.y);

	// ����2�����е��غ�
	Point P2[] = {{0.0, 0.0}, {0.0, 0.0}, {0.0, 0.0}, {0.0, 0.0}};
	int n2 = sizeof(P2) / sizeof(P2[0]);
	ClosestPair result2 = closest(P2, n2);
	printf("����2:\n");
	printf("��̾����� %f ���������ǣ� (%f, %f) (%f, %f)\n",
	       result2.distance, result2.p1.x, result2.p1.y, result2.p2.x, result2.p2.y);

	// ����3�����ڷ��ν���
	Point P3[] = {{0.0, 0.0}, {0.0, 1.0}, {1.0, 0.0}, {1.0, 1.0}};
	int n3 = sizeof(P3) / sizeof(P3[0]);
	ClosestPair result3 = closest(P3, n3);
	printf("����3:\n");
	printf("��̾����� %f ���������ǣ� (%f, %f) (%f, %f)\n",
	       result3.distance, result3.p1.x, result3.p1.y, result3.p2.x, result3.p2.y);

	// ����4������ֲ���
	Point P4[] = {{0.0, 0.0}, {2.0, 3.0}, {5.0, 4.0}, {3.0, 1.0}, {4.0, 0.0}, {1.0, 2.0}};
	int n4 = sizeof(P4) / sizeof(P4[0]);
	ClosestPair result4 = closest(P4, n4);
	printf("����4:\n");
	printf("��̾����� %f ���������ǣ� (%f, %f) (%f, %f)\n",
	       result4.distance, result4.p1.x, result4.p1.y, result4.p2.x, result4.p2.y);

	return 0;
}
