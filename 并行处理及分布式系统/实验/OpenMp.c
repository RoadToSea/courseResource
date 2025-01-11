#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <omp.h>

#define DIM 2  // 二维系统
#define X 0    // x坐标下标
#define Y 1    // y坐标下标

const double G = 6.673e-11;  // 万有引力常数

typedef double vect_t[DIM];  // 向量类型，用于位置、速度等

struct particle_s {
    double m;  // 质量
    vect_t s;  // 位置
    vect_t v;  // 速度
};

void Usage(char *prog_name);
void Get_args(int argc, char *argv[], int *n_threads_p, int *n_p, int *n_steps_p, double *delta_t_p, int *output_freq_p, char *g_i_p);
void Get_init_cond(struct particle_s curr[], int n);
void Gen_init_cond(struct particle_s curr[], int n);
void Output_state(double time, struct particle_s curr[], int n);
void Compute_force(int part, vect_t forces[], struct particle_s curr[], int n);
void Update_part(int part, vect_t forces[], struct particle_s curr[], int n, double delta_t);

int main(int argc, char *argv[]) {
    int n_threads;           // 线程数
    int n;                   // 粒子数量
    int n_steps;             // 时间步数
    double delta_t;          // 时间步长
    int output_freq;         // 输出频率
    char g_i;                // 初始条件生成方式
    struct particle_s *curr; // 粒子数组
    vect_t *forces;          // 力数组
    double start, finish;    // 计时变量

    // 获取命令行参数
    Get_args(argc, argv, &n_threads, &n, &n_steps, &delta_t, &output_freq, &g_i);

    // 分配内存
    curr = (struct particle_s *)malloc(n * sizeof(struct particle_s));
    forces = (vect_t *)malloc(n * sizeof(vect_t));

    if (curr == NULL || forces == NULL) {
        fprintf(stderr, "内存分配失败\n");
        exit(1);
    }

    // 初始化粒子状态
    if (g_i == 'i')
        Get_init_cond(curr, n);
    else
        Gen_init_cond(curr, n);

    // 设置线程数
    omp_set_num_threads(n_threads);

    // 开始计时
    start = omp_get_wtime();

    // 主循环：时间步迭代
    for (int step = 1; step <= n_steps; step++) {
        // 并行计算每个粒子的受力
        #pragma omp parallel for schedule(dynamic, 10)
        for (int i = 0; i < n; i++) {
            Compute_force(i, forces, curr, n);
        }

        // 并行更新每个粒子的位置和速度
        #pragma omp parallel for schedule(dynamic, 10)
        for (int i = 0; i < n; i++) {
            Update_part(i, forces, curr, n, delta_t);
        }

        // 每隔output_freq步输出一次状态
        if (step % output_freq == 0) {
            #pragma omp single
            Output_state(step * delta_t, curr, n);
        }
    }

    // 结束计时
    finish = omp_get_wtime();

    printf("Elapsed time = %e seconds\n", finish - start);

    // 释放内存
    free(curr);
    free(forces);

    return 0;
}

void Usage(char *prog_name) {
    fprintf(stderr, "usage: %s <number of threads> <number of particles> <number of timesteps>\n", prog_name);
    fprintf(stderr, "   <size of timestep> <output frequency>\n");
    fprintf(stderr, "   <g|i>\n");
    fprintf(stderr, "   'g': program should generate init conds\n");
    fprintf(stderr, "   'i': program should get init conds from stdin\n");
    exit(0);
}

void Get_args(int argc, char *argv[], int *n_threads_p, int *n_p, int *n_steps_p, double *delta_t_p, int *output_freq_p, char *g_i_p) {
    if (argc != 7)
        Usage(argv[0]);
    *n_threads_p = strtol(argv[1], NULL, 10);
    *n_p = strtol(argv[2], NULL, 10);
    *n_steps_p = strtol(argv[3], NULL, 10);
    *delta_t_p = strtod(argv[4], NULL);
    *output_freq_p = strtol(argv[5], NULL, 10);
    *g_i_p = argv[6][0];

    if (*n_threads_p <= 0 || *n_p <= 0 || *n_steps_p < 0 || *delta_t_p <= 0)
        Usage(argv[0]);
    if (*g_i_p != 'g' && *g_i_p != 'i')
        Usage(argv[0]);
}

void Get_init_cond(struct particle_s curr[], int n) {
    printf("For each particle, enter (in order):\n");
    printf("   its mass, its x-coord, its y-coord, ");
    printf("its x-velocity, its y-velocity\n");
    for (int part = 0; part < n; part++) {
        scanf("%lf", &curr[part].m);
        scanf("%lf", &curr[part].s[X]);
        scanf("%lf", &curr[part].s[Y]);
        scanf("%lf", &curr[part].v[X]);
        scanf("%lf", &curr[part].v[Y]);
    }
}

void Gen_init_cond(struct particle_s curr[], int n) {
    double mass = 5.0e24;
    double gap = 1.0e5;
    double speed = 3.0e4;

    for (int part = 0; part < n; part++) {
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

void Output_state(double time, struct particle_s curr[], int n) {
    printf("%.2f\n", time);
    for (int part = 0; part < n; part++) {
        printf("%3d %10.3e %10.3e %10.3e %10.3e\n", part, curr[part].s[X], curr[part].s[Y], curr[part].v[X], curr[part].v[Y]);
    }
    printf("\n");
}

void Compute_force(int part, vect_t forces[], struct particle_s curr[], int n) {
    forces[part][X] = forces[part][Y] = 0.0;
    for (int k = 0; k < n; k++) {
        if (k != part) {
            double dx = curr[part].s[X] - curr[k].s[X];
            double dy = curr[part].s[Y] - curr[k].s[Y];
            double dist = sqrt(dx * dx + dy * dy);
            double dist_cubed = dist * dist * dist;
            double force = -G * curr[part].m * curr[k].m / dist_cubed;
            forces[part][X] += force * dx;
            forces[part][Y] += force * dy;
        }
    }
}

void Update_part(int part, vect_t forces[], struct particle_s curr[], int n, double delta_t) {
    double fact = delta_t / curr[part].m;
    curr[part].s[X] += delta_t * curr[part].v[X];
    curr[part].s[Y] += delta_t * curr[part].v[Y];
    curr[part].v[X] += fact * forces[part][X];
    curr[part].v[Y] += fact * forces[part][Y];
}