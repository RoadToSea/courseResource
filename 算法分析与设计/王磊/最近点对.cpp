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
	Point P[] = {{2.1, 3.2}, {12.3, 30.4}, {40.5, 50.6}, {5.7, 1.8}, {12.9, 10.0}, {3.1, 4.2}};
	int n = sizeof(P) / sizeof(P[0]);
	ClosestPair result = closest(P, n);
	printf("��̾����� %f ���������ǣ� (%f, %f) (%f, %f)\n",
	       result.distance, result.p1.x, result.p1.y, result.p2.x, result.p2.y);
	return 0;
}
