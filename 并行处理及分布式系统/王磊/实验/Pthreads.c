/* �ļ�:     pth_nbody_basic.c
 *
 * Ŀ��:  ʹ��Pthreads���л�һ����άn������Ļ����㷨��
 *
 * ����:  gcc -g -Wall -o pth_nbody_basic pth_nbody_basic.c -lm -lpthread
 *        Ҫ�ر���������磬��ʱʱ��������NO_OUTPUT
 *        Ҫ��ȡ��ϸ���������DEBUG
 *        ��Ҫtimer.h
 *
 * ����:  ./pth_nbody_basic <�߳���> <������>
 *           <ʱ�䲽��>  <ʱ�䲽��>
 *           <���Ƶ��> <g|i>
 *           'g': ʹ����������������ɳ�ʼ����
 *           'i': �ӱ�׼�����ȡ��ʼ����
 *          ����Ϊ0.01�������Զ����ɵ����ݡ�
 *
 * ����:  ���������ָ����'g'�����ޡ�
 *          �����'i'������ÿ�����ӵ���������ʼλ�úͳ�ʼ�ٶ�
 * ���:  ������Ƶ��Ϊk����ÿk��ʱ�䲽������λ�ú��ٶ�
 *
 * ��:    ����i��������k�ܵ����������¹�ʽ������
 *
 *    -G m_i m_k (s_i - s_k)/|s_i - s_k|^3
 *
 * ���m_j������j��������s_j����λ������
 * (��ʱ��t)��G�����������������ģ���
 *
 * ע�⣬����k��������i�ܵ�������
 * -(����i��������k�ܵ�����)����ˣ����ǿ��Դ�Լ����������Ĵ�����
 * ��������汾û��������һ�㡣
 *
 * ����:  ����ʹ��ŷ��������
 *
 *    v_i(t+1) = v_i(t) + h v'_i(t)
 *    s_i(t+1) = s_i(t) + h v_i(t)
 *
 * ���v_i(u)��ʱ��uʱ��i�����ӵ��ٶȣ�
 * s_i(u)����λ�á�
 *
 * IPP:  ��6.1.8�ڣ���289ҳ���Ժ�
 *
 */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <pthread.h>
//#include "timer.h"

#include <sys/time.h>

/* ����nowӦ����һ��˫���ȸ�������������ָ��˫���ȸ�������ָ�룩 */
#define GET_TIME(now) { \
		struct timeval t; \
		gettimeofday(&t, NULL); \
		now = t.tv_sec + t.tv_usec/1000000.0; \
	}

#define DIM 2  /* ��άϵͳ */
#define X 0    /* x�����±� */
#define Y 1    /* y�����±� */

const double G = 6.673e-11;  /* ���������� */
/* ��λ��m^3/(kg*s^2)  */

const int BLOCK = 0;         /* ѭ�������Ŀ����  */
const int CYCLIC = 1;        /* ѭ��������ѭ������ */

typedef double vect_t[DIM];  /* λ�õȵ��������� */

struct particle_s {
	double m;  /* ����     */
	vect_t s;  /* λ�� */
	vect_t v;  /* �ٶ� */
};

/* ȫ�ֱ���������ǹ���� */
int thread_count;        /* �߳���                             */
int n;                   /* ������                           */
int n_steps;             /* ʱ�䲽��                          */
double delta_t;          /* ÿ��ʱ�䲽�Ĵ�С                        */
int output_freq;         /* ���֮��Ĳ���                */
struct particle_s *curr; /* ��������״̬������          */
vect_t *forces;          /* ����ÿ�����ӵ����������� */
int b_thread_count = 0;  /* �������ϵ��߳���   */
pthread_mutex_t b_mutex; /* ����ʹ�õĻ�����                         */
pthread_cond_t b_cond_var;  /* ����ʹ�õ���������         */

void Usage(char *prog_name);
void Get_args(int argc, char *argv[], char *g_i_p);
void Get_init_cond(void);
void Gen_init_cond(void);
void Output_state(double time);
void Loop_schedule(int my_rank, int thread_count, int n, int sched,
                   int *first_p, int *last_p, int *incr_p);
void *Thread_work(void *rank);
void Compute_force(int part);
void Update_part(int part);
void Barrier_init(void);
void Barrier(void);
void Barrier_destroy(void);

/*--------------------------------------------------------------------*/
int main(int argc, char *argv[]) {
	char g_i;                   /* _G_enerate��_i_nput��ʼ���� */
	double start, finish;       /* ��ʱ                       */
	long thread;
	pthread_t *thread_handles;

	Get_args(argc, argv, &g_i);
	curr = (particle_s *)malloc(n * sizeof(struct particle_s));
	forces = (vect_t *)malloc(n * sizeof(vect_t));
	if (g_i == 'i')
		Get_init_cond();
	else
		Gen_init_cond();

	thread_handles = (pthread_t *)malloc(thread_count * sizeof(pthread_t));
	Barrier_init();

	GET_TIME(start);
#  ifndef NO_OUTPUT
	Output_state(0.0);
#  endif
	for (thread = 0; thread < thread_count; thread++)
		pthread_create(&thread_handles[thread], NULL,
		               Thread_work, (void *) thread);

	for (thread = 0; thread < thread_count; thread++)
		pthread_join(thread_handles[thread], NULL);

	GET_TIME(finish);
	printf("Elapsed time = %e seconds\n", finish - start);

	Barrier_destroy();
	free(thread_handles);
	free(curr);
	free(forces);
	return 0;
}  /* main */

/*---------------------------------------------------------------------
 * ����: Usage
 * Ŀ��:  ��ӡ������ָ��˳�
 * �������:
 *     prog_name:  �������ƣ������������м���
 */
void Usage(char *prog_name) {
	fprintf(stderr, "usage: %s <�߳���> <������>\n",
	        prog_name);
	fprintf(stderr, "   <ʱ�䲽��>  <ʱ�䲽��>\n");
	fprintf(stderr, "   <���Ƶ��> <g|i>\n");
	fprintf(stderr, "   'g': ����Ӧ���ɳ�ʼ����\n");
	fprintf(stderr, "   'i': ����Ӧ�ӱ�׼�����ȡ��ʼ����\n");

	exit(0);
}  /* Usage */

/*---------------------------------------------------------------------
 * ����:  Get_args
 * Ŀ��:  ��ȡ�����в���
 * �������:
 *    argc:           �����в�������
 *    argv:           �����в���
 * ȫ�ֱ�����ȫ�������:
 *    thread_count:   �߳���
 *    n:              ������
 *    n_steps:        ʱ�䲽��
 *    delta_t:        ÿ��ʱ�䲽�Ĵ�С
 *    output_freq:    ���֮���ʱ�䲽��
 * �������:
 *    g_i_p:          ָ���ַ���ָ�룬�����ʼ����Ӧ�ɳ������ɣ���Ϊ'g'��
 *                    ���Ӧ�ӱ�׼�����ȡ����Ϊ'i'
 */
void Get_args(int argc, char *argv[], char *g_i_p) {
	if (argc != 7)
		Usage(argv[0]);
	thread_count = strtol(argv[1], NULL, 10);
	n = strtol(argv[2], NULL, 10);
	n_steps = strtol(argv[3], NULL, 10);
	delta_t = strtod(argv[4], NULL);
	output_freq = strtol(argv[5], NULL, 10);
	*g_i_p = argv[6][0];

	if (thread_count <= 0 || n <= 0 || n_steps < 0 ||
	        delta_t <= 0)
		Usage(argv[0]);
	if (*g_i_p != 'g' && *g_i_p != 'i')
		Usage(argv[0]);

#  ifdef DEBUG
	printf("thread_count = %d\n", thread_count);
	printf("n = %d\n", n);
	printf("n_steps = %d\n", n_steps);
	printf("delta_t = %e\n", delta_t);
	printf("output_freq = %d\n", output_freq);
	printf("g_i = %c\n", *g_i_p);
#  endif
}  /* Get_args */

/*---------------------------------------------------------------------
 * ����:  Get_init_cond
 * Ŀ��:  ��ȡ��ʼ������ÿ�����ӵ�������λ�ú��ٶ�
 * ȫ�ֱ���:
 *    n (in):      ������
 *    curr (out):  ����n���ṹ�����飬ÿ���ṹ�洢���ӵ���������������
 *      λ�ã����������ٶȣ�������
 */
void Get_init_cond(void) {
	int part;

	printf("����ÿ�����ӣ���˳�����룺\n");
	printf("   ����������x���꣬��y���꣬ ");
	printf("��x�ٶȣ���y�ٶ�\n");
	for (part = 0; part < n; part++) {
		scanf("%lf", &curr[part].m);
		scanf("%lf", &curr[part].s[X]);
		scanf("%lf", &curr[part].s[Y]);
		scanf("%lf", &curr[part].v[X]);
		scanf("%lf", &curr[part].v[Y]);
	}
}  /* Get_init_cond */

/*---------------------------------------------------------------------
 * ����:  Gen_init_cond
 * Ŀ��:  ���ɳ�ʼ������ÿ�����ӵ�������λ�ú��ٶ�
 * ȫ�ֱ���:
 *    n (in):      �����������룩
 *    curr (out):  ����n���ṹ�����飬ÿ���ṹ�洢���ӵ���������������
 *      λ�ã����������ٶȣ�������
 *
 * ע��:      ��ʼ�������������ӷ�����
 *            �Ǹ�x���ϵĵȼ��λ�ã�
 *            ������ͬ����������ͬ�ĳ�ʼ�ٶ�
 *            ƽ����y�ᡣȻ����һЩ
 *            �ٶ�����y����
 *            һЩ�Ǹ��ġ�
 */
void Gen_init_cond(void) {
	int part;
	double mass = 5.0e24;
	double gap = 1.0e5;
	double speed = 3.0e4;

	srandom(1);
	for (part = 0; part < n; part++) {
		curr[part].m = mass;
		curr[part].s[X] = part * gap;
		curr[part].s[Y] = 0.0;
		curr[part].v[X] = 0.0;
//    if (random()/((double) RAND_MAX) >= 0.5)
		if (part % 2 == 0)
			curr[part].v[Y] = speed;
		else
			curr[part].v[Y] = -speed;
	}
}  /* Gen_init_cond */

/*---------------------------------------------------------------------
 * ����:  Loop_sched
 * Ŀ��:  ���ؿ��ѭ�����ȵĲ���
 *           ����forѭ��
 * �������:
 *    my_rank:       �����̵߳�����
 *    thread_count:  �߳���
 *    n:             ѭ����������
 *    sched:         ���ȣ�BLOCK��CYCLIC
 * �������:
 *    first_p:       ָ���һ��ѭ��������ָ��
 *    last_p:        ָ��������һ��������ֵ��ָ��
 *    incr_p:        ѭ������
 */
void Loop_schedule(int my_rank, int thread_count, int n, int sched,

                   int *first_p, int *last_p, int *incr_p) {
	if (sched == CYCLIC) {
		*first_p = my_rank;
		*last_p = n;
		*incr_p = thread_count;
	} else {  /* sched == BLOCK */
		int quotient = n / thread_count;
		int remainder = n % thread_count;
		int my_iters;
		*incr_p = 1;
		if (my_rank < remainder) {
			my_iters = quotient + 1;
			*first_p = my_rank * my_iters;
		} else {
			my_iters = quotient;
			*first_p = my_rank * my_iters + remainder;
		}
		*last_p = *first_p + my_iters;
	}
}  /* Loop_schedule */

/*---------------------------------------------------------------------
 * ����:  Thread_work
 * Ŀ��:  ִ�е����̶߳��ҵ�����λ�ú��ٶȵĹ��ס�
 * �������:
 *    rank:   �̵߳�������0, 1, . . . , thread_count-1��
 * ȫ�ֱ���:
 *    thread_count (in):
 *
 */
void *Thread_work(void *rank) {
	long my_rank = (long) rank;
	int step;    /* ��ǰ����      */
	int part;    /* ��ǰ����  */
	double t;    /* ��ǰʱ��      */
	int first;   /* �ҵĵ�һ������ */
	int last;    /* �ҵ����һ������  */
	int incr;    /* ѭ������    */

	Loop_schedule(my_rank, thread_count, n, BLOCK, &first, &last, &incr);
	for (step = 1; step <= n_steps; step++) {
		t = step * delta_t;
		/* ����n-1���ڵ��ú����������
		 * Compute_force(n-2, . . .) */
		for (part = first; part < last; part += incr)
			Compute_force(part);
		Barrier();
		for (part = first; part < last; part += incr)
			Update_part(part);
		Barrier();
#     ifndef NO_OUTPUT
		if (step % output_freq == 0 && my_rank == 0) {
			Output_state(t);
		}
#     endif
	}  /* for step */

	return NULL;
}  /* Thread_work */

/*---------------------------------------------------------------------
 * ����:  Output_state
 * Ŀ��:  ��ӡϵͳ�ĵ�ǰ״̬
 * �������:
 *    t:      ��ǰʱ��
 * ȫ�ֱ�����ȫ�����룩:
 *    curr:   ����n��Ԫ�ص����飬curr[i]�洢��i�����ӵ�״̬��������
 *            λ�ú��ٶȣ�
 *    n:      ������
 */
void Output_state(double time) {
	int part;
	printf("%.2f\n", time);
	for (part = 0; part < n; part++) {
//    printf("%.3e ", curr[part].m);
		printf("%3d %10.3e ", part, curr[part].s[X]);
		printf("  %10.3e ", curr[part].s[Y]);
		printf("  %10.3e ", curr[part].v[X]);
		printf("  %10.3e\n", curr[part].v[Y]);
	}
	printf("\n");
}  /* Output_state */

/*---------------------------------------------------------------------
 * ����:  Compute_force
 * Ŀ��:  ��������part�����������
 *           �汾�����öԳ��ԣ�����i��������k�ܵ�������= -������k��������i�ܵ���������
 * �������:
 *    part:   ���Ǽ�������������
 * ȫ�ֱ���:
 *    curr (in):  ϵͳ�ĵ�ǰ״̬��curr[i]�洢��i�����ӵ�������
 *      λ�ú��ٶ�
 *    n (in):     ������
 *    forces (out): forces[i]�洢��i�����ӵ�����
 *
 * ע��: �������ʹ�������������
 * ����i��������k�ܵ����������¹�ʽ������
 *
 *    m_i m_k (s_k - s_i)/|s_k - s_i|^2
 *
 * ���m_j������j��������s_k����λ������
 * (��ʱ��t)��
 */
void Compute_force(int part) {
	int k;
	double mg;
	vect_t f_part_k;
	double len, len_3, fact;

#  ifdef DEBUG
	printf("��ǰ����%d������ = (%.3e, %.3e)\n",
	       part, forces[part][X], forces[part][Y]);
#  endif
	forces[part][X] = forces[part][Y] = 0.0;
	for (k = 0; k < n; k++) {
		if (k != part) {
			/* ��������part��������k�ܵ����� */
			f_part_k[X] = curr[part].s[X] - curr[k].s[X];
			f_part_k[Y] = curr[part].s[Y] - curr[k].s[Y];
			len = sqrt(f_part_k[X] * f_part_k[X] + f_part_k[Y] * f_part_k[Y]);
			len_3 = len * len * len;
			mg = -G * curr[part].m * curr[k].m;
			fact = mg / len_3;
			f_part_k[X] *= fact;
			f_part_k[Y] *= fact;
#        ifdef DEBUG
			printf("����%d��������%d�ܵ����� = (%.3e, %.3e)\n",
			       part, k, f_part_k[X], f_part_k[Y]);
#        endif

			/* ������ӵ������� */
			forces[part][X] += f_part_k[X];
			forces[part][Y] += f_part_k[Y];

			/* ���öԳ��Ը�������k��������part�ܵ����� */
			forces[k][X] -= f_part_k[X];
			forces[k][Y] -= f_part_k[Y];
		}
	}
}  /* Compute_force */

/*---------------------------------------------------------------------
 * ����:  Update_part
 * Ŀ��:  ��������part���ٶȺ�λ��
 * �������:
 *    part:    ����Ҫ���µ�����
 * ȫ�ֱ���:
 *    forces (in):   forces[i]�洢��i�����ӵ�����
 *    n (in):        ������
 *    curr (in/out): curr[i]�洢��i�����ӵ�������λ�ú��ٶ�
 *
 * ע��:  ����汾ʹ��ŷ�����������ٶ�
 *    ��λ�á�
 */
void Update_part(int part) {
	double fact = delta_t / curr[part].m;

#  ifdef DEBUG
	printf("����%d֮ǰ��\n", part);
	printf("   λ��  = (%.3e, %.3e)\n", curr[part].s[X], curr[part].s[Y]);
	printf("   �ٶ�  = (%.3e, %.3e)\n", curr[part].v[X], curr[part].v[Y]);
	printf("   ����  = (%.3e, %.3e)\n", forces[part][X], forces[part][Y]);
#  endif
	curr[part].s[X] += delta_t *curr[part].v[X];
	curr[part].s[Y] += delta_t *curr[part].v[Y];
	curr[part].v[X] += fact * forces[part][X];
	curr[part].v[Y] += fact * forces[part][Y];
#  ifdef DEBUG
	printf("����%d��λ�� = (%.3e, %.3e)���ٶ� = (%.3e,%.3e)\n",
	       part, curr[part].s[X], curr[part].s[Y],
	       curr[part].v[X], curr[part].v[Y]);
#  endif
// curr[part].s[X] += delta_t * curr[part].v[X];
// curr[part].s[Y] += delta_t * curr[part].v[Y];
}  /* Update_part */

/*---------------------------------------------------------------------
 * ����:    Barrier_init
 * Ŀ��:    ��ʼ��������������ݽṹ
 * ȫ�ֱ�����ȫ�������:
 *    b_thread_count:  �����е��߳���
 *    b_mutex:         ����ʹ�õĻ�����
 *    b_cond_var:      ����ʹ�õ���������
 */
void Barrier_init(void) {
	b_thread_count = 0;
	pthread_mutex_init(&b_mutex, NULL);
	pthread_cond_init(&b_cond_var, NULL);
}  /* Barrier_init */

/*---------------------------------------------------------------------
 * ����:    Barrier
 * Ŀ��:    ������ֱ�������̶߳���������
 * ȫ�ֱ���:
 *    thread_count (in):       ���߳���
 *    b_thread_count (in/out): �����е��߳���
 *    b_mutex (in/out):        ����ʹ�õĻ�����
 *    b_cond_var (in/out):     ����ʹ�õ���������
 */
void Barrier(void) {
	pthread_mutex_lock(&b_mutex);
	b_thread_count++;
	if (b_thread_count == thread_count) {
		b_thread_count = 0;
		pthread_cond_broadcast(&b_cond_var);
	} else {
		// �ȴ�������������ʹ�߳����ߡ�
		//    ���ȴ�����whileѭ���У��Է�ĳЩ����
		// �¼������̡߳�
		while (pthread_cond_wait(&b_cond_var, &b_mutex) != 0);
		// �������ڴ˵�����������
	}
	pthread_mutex_unlock(&b_mutex);
}  /* Barrier */

/*---------------------------------------------------------------------
 * ����:    Barrier_destroy
 * Ŀ��:    ����������������ݽṹ
 * ȫ�ֱ�����ȫ�������:
 *    b_mutex:         ����ʹ�õĻ�����
 *    b_cond_var:      ����ʹ�õ���������
 */
void Barrier_destroy(void) {
	pthread_mutex_destroy(&b_mutex);
	pthread_cond_destroy(&b_cond_var);
}  /* Barrier_destroy */
