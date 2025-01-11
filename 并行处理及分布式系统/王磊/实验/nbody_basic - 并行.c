#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <sys/time.h>
#include <omp.h>

/* 获取当前时间的宏定义 */
#define GET_TIME(now) { \
		struct timeval t; \
		gettimeofday(&t, NULL); \
		now = t.tv_sec + t.tv_usec / 1000000.0; \
	}

#define DIM 2  /* 二维系统 */
#define X 0    /* x坐标的下标 */
#define Y 1    /* y坐标的下标 */

const double G = 6.673e-11;  /* 引力常数 */
/* 单位是 m^3/(kg*s^2) */

/* 定义粒子结构体 */
typedef double vect_t[DIM];  /* 向量类型，表示位置、速度等 */

struct particle_s {
	double m;  /* 质量 */
	vect_t s;  /* 位置 */
	vect_t v;  /* 速度 */
};

/* 函数声明 */
void Usage(char *prog_name);
void Get_args(int argc, char *argv[], int *n_p, int *n_steps_p,
              double *delta_t_p, int *output_freq_p, char *g_i_p);
void Get_init_cond(struct particle_s curr[], int n);
void Gen_init_cond(struct particle_s curr[], int n);
void Output_state(double time, struct particle_s curr[], int n);
void Compute_force(int part, vect_t forces[], struct particle_s curr[],
                   int n);
void Update_part(int part, vect_t forces[], struct particle_s curr[],
                 int n, double delta_t);
void Compute_energy(struct particle_s curr[], int n, double *kin_en_p,
                    double *pot_en_p);

int main(int argc, char *argv[]) {
	int n;                      /* 粒子数量 */
	int n_steps;                /* 时间步数 */
	int step;                   /* 当前步数 */
	int part;                   /* 当前粒子 */
	int output_freq;            /* 输出频率 */
	double delta_t;             /* 时间步长 */
	double t;                   /* 当前时间 */
	struct particle_s *curr;    /* 当前系统状态 */
	vect_t *forces;             /* 每个粒子上的力 */
	char g_i;                   /* 'g'表示生成初始条件，'i'表示从标准输入读取 */

#  ifdef COMPUTE_ENERGY
	double kinetic_energy, potential_energy;
#  endif
	double start, finish;       /* 用于计时 */

	/* 获取命令行参数 */
	Get_args(argc, argv, &n, &n_steps, &delta_t, &output_freq, &g_i);
	curr = malloc(n * sizeof(struct particle_s));
	forces = malloc(n * sizeof(vect_t));
	if (g_i == 'i')
		Get_init_cond(curr, n);
	else
		Gen_init_cond(curr, n);

	GET_TIME(start);

#  ifdef COMPUTE_ENERGY
	/* 计算并输出初始能量 */
	Compute_energy(curr, n, &kinetic_energy, &potential_energy);
	printf("初始状态: PE = %e, KE = %e, 总能量 = %e\n",
	       potential_energy, kinetic_energy, kinetic_energy + potential_energy);
#  endif

#  ifndef NO_OUTPUT
	/* 输出初始状态 */
	Output_state(0, curr, n);
#  endif

	/* 主时间循环 */
	for (step = 1; step <= n_steps; step++) {
		t = step * delta_t;

		/* 并行计算每个粒子受到的力 */
		#pragma omp parallel for
		for (part = 0; part < n; part++) {
			Compute_force(part, forces, curr, n);
		}

		/* 并行更新每个粒子的状态 */
		#pragma omp parallel for
		for (part = 0; part < n; part++) {
			Update_part(part, forces, curr, n, delta_t);
		}

#     ifdef COMPUTE_ENERGY
		/* 计算并输出当前能量 */
		Compute_energy(curr, n, &kinetic_energy, &potential_energy);
		printf("第 %d 步: PE = %e, KE = %e, 总能量 = %e\n",
		       step, potential_energy, kinetic_energy, kinetic_energy + potential_energy);
#     endif

#     ifndef NO_OUTPUT
		/* 每隔指定步数输出一次状态 */
		if (step % output_freq == 0)
			Output_state(t, curr, n);
#     endif
	}

	GET_TIME(finish);
	printf("总耗时 = %e 秒\n", finish - start);

	free(curr);
	free(forces);
	return 0;
}

/*---------------------------------------------------------------------
 * 用法提示函数
 * 打印命令行参数使用说明并退出
 */
void Usage(char *prog_name) {
	fprintf(stderr, "用法: %s <粒子数量> <时间步数>\n", prog_name);
	fprintf(stderr, "   <时间步长> <输出频率>\n");
	fprintf(stderr, "   <g|i>\n");
	fprintf(stderr, "   'g': 程序自动生成初始条件\n");
	fprintf(stderr, "   'i': 从标准输入读取初始条件\n");

	exit(0);
}

/*---------------------------------------------------------------------
 * 获取命令行参数
 */
void Get_args(int argc, char *argv[], int *n_p, int *n_steps_p,

              double *delta_t_p, int *output_freq_p, char *g_i_p) {
	if (argc != 6)
		Usage(argv[0]);
	*n_p = strtol(argv[1], NULL, 10);
	*n_steps_p = strtol(argv[2], NULL, 10);
	*delta_t_p = strtod(argv[3], NULL);
	*output_freq_p = strtol(argv[4], NULL, 10);
	*g_i_p = argv[5][0];

	if (*n_p <= 0 || *n_steps_p < 0 || *delta_t_p <= 0)
		Usage(argv[0]);
	if (*g_i_p != 'g' && *g_i_p != 'i')
		Usage(argv[0]);
}

/*---------------------------------------------------------------------
 * 从标准输入读取粒子的初始条件（质量、位置、速度）
 */
void Get_init_cond(struct particle_s curr[], int n) {
	int part;

	printf("请输入每个粒子的质量、位置和速度：\n");
	for (part = 0; part < n; part++) {
		scanf("%lf", &curr[part].m);
		scanf("%lf", &curr[part].s[X]);
		scanf("%lf", &curr[part].s[Y]);
		scanf("%lf", &curr[part].v[X]);
		scanf("%lf", &curr[part].v[Y]);
	}
}

/*---------------------------------------------------------------------
 * 生成粒子的初始条件（均匀分布，速度随机）
 */
void Gen_init_cond(struct particle_s curr[], int n) {
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
		if (part % 2 == 0)
			curr[part].v[Y] = speed;
		else
			curr[part].v[Y] = -speed;
	}
}

/*---------------------------------------------------------------------
 * 输出系统当前状态
 */
void Output_state(double time, struct particle_s curr[], int n) {
	int part;
	printf("%.2f\n", time);
	for (part = 0; part < n; part++) {
		printf("%3d %10.3e ", part, curr[part].s[X]);
		printf("  %10.3e ", curr[part].s[Y]);
		printf("  %10.3e ", curr[part].v[X]);
		printf("  %10.3e\n", curr[part].v[Y]);
	}
	printf("\n");
}

/*---------------------------------------------------------------------
 * 计算粒子受到的引力
 */
void Compute_force(int part, vect_t forces[], struct particle_s curr[],

                   int n) {
	int k;
	double mg;
	vect_t f_part_k;
	double len, len_3, fact;

	forces[part][X] = forces[part][Y] = 0.0;
	for (k = 0; k < n; k++) {
		if (k != part) {
			f_part_k[X] = curr[part].s[X] - curr[k].s[X];
			f_part_k[Y] = curr[part].s[Y] - curr[k].s[Y];
			len = sqrt(f_part_k[X] * f_part_k[X] + f_part_k[Y] * f_part_k[Y]);
			len_3 = len * len * len;
			fact = G * curr[part].m * curr[k].m / len_3;
			forces[part][X] += fact * f_part_k[X];
			forces[part][Y] += fact * f_part_k[Y];
		}
	}
}

/*---------------------------------------------------------------------
 * 更新粒子的状态（位置和速度）
 */
void Update_part(int part, vect_t forces[], struct particle_s curr[],

                 int n, double delta_t) {
	curr[part].v[X] += forces[part][X] * delta_t / curr[part].m;
	curr[part].v[Y] += forces[part][Y] * delta_t / curr[part].m;
	curr[part].s[X] += curr[part].v[X] * delta_t;
	curr[part].s[Y] += curr[part].v[Y] * delta_t;
}

/*---------------------------------------------------------------------
 * 计算系统的动能和势能
 */
void Compute_energy(struct particle_s curr[], int n, double *kin_en_p,
                    double *pot_en_p) {
	int part, k;
	double len, len_3;
	*kin_en_p = 0.0;
	*pot_en_p = 0.0;

	for (part = 0; part < n; part++) {
		*kin_en_p += 0.5 * curr[part].m * (curr[part].v[X] * curr[part].v[X] +
		                                   curr[part].v[Y] * curr[part].v[Y]);
		for (k = part + 1; k < n; k++) {
			len = sqrt((curr[part].s[X] - curr[k].s[X]) * (curr[part].s[X] - curr[k].s[X]) +
			           (curr[part].s[Y] - curr[k].s[Y]) * (curr[part].s[Y] - curr[k].s[Y]));
			len_3 = len * len * len;
			*pot_en_p -= G * curr[part].m * curr[k].m / len;
		}
	}
}
