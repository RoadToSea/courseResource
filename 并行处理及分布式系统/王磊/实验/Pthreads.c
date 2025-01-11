/* 文件:     pth_nbody_basic.c
 *
 * 目的:  使用Pthreads并行化一个二维n体问题的基本算法。
 *
 * 编译:  gcc -g -Wall -o pth_nbody_basic pth_nbody_basic.c -lm -lpthread
 *        要关闭输出（例如，计时时），定义NO_OUTPUT
 *        要获取详细输出，定义DEBUG
 *        需要timer.h
 *
 * 运行:  ./pth_nbody_basic <线程数> <粒子数>
 *           <时间步数>  <时间步长>
 *           <输出频率> <g|i>
 *           'g': 使用随机数生成器生成初始条件
 *           'i': 从标准输入读取初始条件
 *          步长为0.01适用于自动生成的数据。
 *
 * 输入:  如果命令行指定了'g'，则无。
 *          如果是'i'，则是每个粒子的质量、初始位置和初始速度
 * 输出:  如果输出频率为k，则每k个时间步的粒子位置和速度
 *
 * 力:    粒子i由于粒子k受到的力由以下公式给出：
 *
 *    -G m_i m_k (s_i - s_k)/|s_i - s_k|^3
 *
 * 这里，m_j是粒子j的质量，s_j是其位置向量
 * (在时间t)，G是引力常数（见下文）。
 *
 * 注意，粒子k由于粒子i受到的力是
 * -(粒子i由于粒子k受到的力)。因此，我们可以大约减半力计算的次数，
 * 尽管这个版本没有利用这一点。
 *
 * 积分:  我们使用欧拉方法：
 *
 *    v_i(t+1) = v_i(t) + h v'_i(t)
 *    s_i(t+1) = s_i(t) + h v_i(t)
 *
 * 这里，v_i(u)是时间u时第i个粒子的速度，
 * s_i(u)是其位置。
 *
 * IPP:  第6.1.8节（第289页及以后）
 *
 */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <pthread.h>
//#include "timer.h"

#include <sys/time.h>

/* 参数now应该是一个双精度浮点数（而不是指向双精度浮点数的指针） */
#define GET_TIME(now) { \
		struct timeval t; \
		gettimeofday(&t, NULL); \
		now = t.tv_sec + t.tv_usec/1000000.0; \
	}

#define DIM 2  /* 二维系统 */
#define X 0    /* x坐标下标 */
#define Y 1    /* y坐标下标 */

const double G = 6.673e-11;  /* 引力常数。 */
/* 单位是m^3/(kg*s^2)  */

const int BLOCK = 0;         /* 循环迭代的块分区  */
const int CYCLIC = 1;        /* 循环迭代的循环分区 */

typedef double vect_t[DIM];  /* 位置等的向量类型 */

struct particle_s {
	double m;  /* 质量     */
	vect_t s;  /* 位置 */
	vect_t v;  /* 速度 */
};

/* 全局变量，因此是共享的 */
int thread_count;        /* 线程数                             */
int n;                   /* 粒子数                           */
int n_steps;             /* 时间步数                          */
double delta_t;          /* 每个时间步的大小                        */
int output_freq;         /* 输出之间的步数                */
struct particle_s *curr; /* 包含粒子状态的数组          */
vect_t *forces;          /* 包含每个粒子的总力的数组 */
int b_thread_count = 0;  /* 进入屏障的线程数   */
pthread_mutex_t b_mutex; /* 屏障使用的互斥锁                         */
pthread_cond_t b_cond_var;  /* 屏障使用的条件变量         */

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
	char g_i;                   /* _G_enerate或_i_nput初始条件 */
	double start, finish;       /* 计时                       */
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
 * 函数: Usage
 * 目的:  打印命令行指令并退出
 * 输入参数:
 *     prog_name:  程序名称，如在命令行中键入
 */
void Usage(char *prog_name) {
	fprintf(stderr, "usage: %s <线程数> <粒子数>\n",
	        prog_name);
	fprintf(stderr, "   <时间步数>  <时间步长>\n");
	fprintf(stderr, "   <输出频率> <g|i>\n");
	fprintf(stderr, "   'g': 程序应生成初始条件\n");
	fprintf(stderr, "   'i': 程序应从标准输入获取初始条件\n");

	exit(0);
}  /* Usage */

/*---------------------------------------------------------------------
 * 函数:  Get_args
 * 目的:  获取命令行参数
 * 输入参数:
 *    argc:           命令行参数数量
 *    argv:           命令行参数
 * 全局变量（全部输出）:
 *    thread_count:   线程数
 *    n:              粒子数
 *    n_steps:        时间步数
 *    delta_t:        每个时间步的大小
 *    output_freq:    输出之间的时间步数
 * 输出参数:
 *    g_i_p:          指向字符的指针，如果初始条件应由程序生成，则为'g'，
 *                    如果应从标准输入读取，则为'i'
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
 * 函数:  Get_init_cond
 * 目的:  读取初始条件：每个粒子的质量、位置和速度
 * 全局变量:
 *    n (in):      粒子数
 *    curr (out):  包含n个结构的数组，每个结构存储粒子的质量（标量）、
 *      位置（向量）和速度（向量）
 */
void Get_init_cond(void) {
	int part;

	printf("对于每个粒子，按顺序输入：\n");
	printf("   其质量，其x坐标，其y坐标， ");
	printf("其x速度，其y速度\n");
	for (part = 0; part < n; part++) {
		scanf("%lf", &curr[part].m);
		scanf("%lf", &curr[part].s[X]);
		scanf("%lf", &curr[part].s[Y]);
		scanf("%lf", &curr[part].v[X]);
		scanf("%lf", &curr[part].v[Y]);
	}
}  /* Get_init_cond */

/*---------------------------------------------------------------------
 * 函数:  Gen_init_cond
 * 目的:  生成初始条件：每个粒子的质量、位置和速度
 * 全局变量:
 *    n (in):      粒子数（输入）
 *    curr (out):  包含n个结构的数组，每个结构存储粒子的质量（标量）、
 *      位置（向量）和速度（向量）
 *
 * 注意:      初始条件将所有粒子放置在
 *            非负x轴上的等间距位置，
 *            具有相同的质量和相同的初始速度
 *            平行于y轴。然而，一些
 *            速度在正y方向，
 *            一些是负的。
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
 * 函数:  Loop_sched
 * 目的:  返回块或循环调度的参数
 *           用于for循环
 * 输入参数:
 *    my_rank:       调用线程的排名
 *    thread_count:  线程数
 *    n:             循环迭代次数
 *    sched:         调度：BLOCK或CYCLIC
 * 输出参数:
 *    first_p:       指向第一个循环索引的指针
 *    last_p:        指向大于最后一个索引的值的指针
 *    incr_p:        循环增量
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
 * 函数:  Thread_work
 * 目的:  执行单个线程对找到粒子位置和速度的贡献。
 * 输入参数:
 *    rank:   线程的排名（0, 1, . . . , thread_count-1）
 * 全局变量:
 *    thread_count (in):
 *
 */
void *Thread_work(void *rank) {
	long my_rank = (long) rank;
	int step;    /* 当前步骤      */
	int part;    /* 当前粒子  */
	double t;    /* 当前时间      */
	int first;   /* 我的第一个粒子 */
	int last;    /* 我的最后一个粒子  */
	int incr;    /* 循环增量    */

	Loop_schedule(my_rank, thread_count, n, BLOCK, &first, &last, &incr);
	for (step = 1; step <= n_steps; step++) {
		t = step * delta_t;
		/* 粒子n-1将在调用后计算所有力
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
 * 函数:  Output_state
 * 目的:  打印系统的当前状态
 * 输入参数:
 *    t:      当前时间
 * 全局变量（全部输入）:
 *    curr:   包含n个元素的数组，curr[i]存储第i个粒子的状态（质量、
 *            位置和速度）
 *    n:      粒子数
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
 * 函数:  Compute_force
 * 目的:  计算粒子part的总力。这个
 *           版本不利用对称性（粒子i由于粒子k受到的力）= -（粒子k由于粒子i受到的力）。
 * 输入参数:
 *    part:   我们计算总力的粒子
 * 全局变量:
 *    curr (in):  系统的当前状态：curr[i]存储第i个粒子的质量、
 *      位置和速度
 *    n (in):     粒子数
 *    forces (out): forces[i]存储第i个粒子的总力
 *
 * 注意: 这个函数使用引力力。因此
 * 粒子i由于粒子k受到的力由以下公式给出：
 *
 *    m_i m_k (s_k - s_i)/|s_k - s_i|^2
 *
 * 这里，m_j是粒子j的质量，s_k是其位置向量
 * (在时间t)。
 */
void Compute_force(int part) {
	int k;
	double mg;
	vect_t f_part_k;
	double len, len_3, fact;

#  ifdef DEBUG
	printf("当前粒子%d的总力 = (%.3e, %.3e)\n",
	       part, forces[part][X], forces[part][Y]);
#  endif
	forces[part][X] = forces[part][Y] = 0.0;
	for (k = 0; k < n; k++) {
		if (k != part) {
			/* 计算粒子part由于粒子k受到的力 */
			f_part_k[X] = curr[part].s[X] - curr[k].s[X];
			f_part_k[Y] = curr[part].s[Y] - curr[k].s[Y];
			len = sqrt(f_part_k[X] * f_part_k[X] + f_part_k[Y] * f_part_k[Y]);
			len_3 = len * len * len;
			mg = -G * curr[part].m * curr[k].m;
			fact = mg / len_3;
			f_part_k[X] *= fact;
			f_part_k[Y] *= fact;
#        ifdef DEBUG
			printf("粒子%d由于粒子%d受到的力 = (%.3e, %.3e)\n",
			       part, k, f_part_k[X], f_part_k[Y]);
#        endif

			/* 将力添加到总力中 */
			forces[part][X] += f_part_k[X];
			forces[part][Y] += f_part_k[Y];

			/* 利用对称性更新粒子k由于粒子part受到的力 */
			forces[k][X] -= f_part_k[X];
			forces[k][Y] -= f_part_k[Y];
		}
	}
}  /* Compute_force */

/*---------------------------------------------------------------------
 * 函数:  Update_part
 * 目的:  更新粒子part的速度和位置
 * 输入参数:
 *    part:    我们要更新的粒子
 * 全局变量:
 *    forces (in):   forces[i]存储第i个粒子的总力
 *    n (in):        粒子数
 *    curr (in/out): curr[i]存储第i个粒子的质量、位置和速度
 *
 * 注意:  这个版本使用欧拉方法更新速度
 *    和位置。
 */
void Update_part(int part) {
	double fact = delta_t / curr[part].m;

#  ifdef DEBUG
	printf("更新%d之前：\n", part);
	printf("   位置  = (%.3e, %.3e)\n", curr[part].s[X], curr[part].s[Y]);
	printf("   速度  = (%.3e, %.3e)\n", curr[part].v[X], curr[part].v[Y]);
	printf("   净力  = (%.3e, %.3e)\n", forces[part][X], forces[part][Y]);
#  endif
	curr[part].s[X] += delta_t *curr[part].v[X];
	curr[part].s[Y] += delta_t *curr[part].v[Y];
	curr[part].v[X] += fact * forces[part][X];
	curr[part].v[Y] += fact * forces[part][Y];
#  ifdef DEBUG
	printf("粒子%d的位置 = (%.3e, %.3e)，速度 = (%.3e,%.3e)\n",
	       part, curr[part].s[X], curr[part].s[Y],
	       curr[part].v[X], curr[part].v[Y]);
#  endif
// curr[part].s[X] += delta_t * curr[part].v[X];
// curr[part].s[Y] += delta_t * curr[part].v[Y];
}  /* Update_part */

/*---------------------------------------------------------------------
 * 函数:    Barrier_init
 * 目的:    初始化屏障所需的数据结构
 * 全局变量（全部输出）:
 *    b_thread_count:  屏障中的线程数
 *    b_mutex:         屏障使用的互斥锁
 *    b_cond_var:      屏障使用的条件变量
 */
void Barrier_init(void) {
	b_thread_count = 0;
	pthread_mutex_init(&b_mutex, NULL);
	pthread_cond_init(&b_cond_var, NULL);
}  /* Barrier_init */

/*---------------------------------------------------------------------
 * 函数:    Barrier
 * 目的:    阻塞，直到所有线程都进入屏障
 * 全局变量:
 *    thread_count (in):       总线程数
 *    b_thread_count (in/out): 屏障中的线程数
 *    b_mutex (in/out):        屏障使用的互斥锁
 *    b_cond_var (in/out):     屏障使用的条件变量
 */
void Barrier(void) {
	pthread_mutex_lock(&b_mutex);
	b_thread_count++;
	if (b_thread_count == thread_count) {
		b_thread_count = 0;
		pthread_cond_broadcast(&b_cond_var);
	} else {
		// 等待解锁互斥锁并使线程休眠。
		//    将等待放在while循环中，以防某些其他
		// 事件唤醒线程。
		while (pthread_cond_wait(&b_cond_var, &b_mutex) != 0);
		// 互斥锁在此点重新锁定。
	}
	pthread_mutex_unlock(&b_mutex);
}  /* Barrier */

/*---------------------------------------------------------------------
 * 函数:    Barrier_destroy
 * 目的:    销毁屏障所需的数据结构
 * 全局变量（全部输出）:
 *    b_mutex:         屏障使用的互斥锁
 *    b_cond_var:      屏障使用的条件变量
 */
void Barrier_destroy(void) {
	pthread_mutex_destroy(&b_mutex);
	pthread_cond_destroy(&b_cond_var);
}  /* Barrier_destroy */
