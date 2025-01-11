#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
//#include "timer.h"

#include <sys/time.h>
#include <pthread.h>

/* The argument now should be a double (not a pointer to a double) */
#define GET_TIME(now) { \
		struct timeval t; \
		gettimeofday(&t, NULL); \
		now = t.tv_sec + t.tv_usec/1000000.0; \
	}

#define DIM 2  /* Two-dimensional system */
#define X 0    /* x-coordinate subscript */
#define Y 1    /* y-coordinate subscript */

const double G = 6.673e-11;  /* Gravitational constant. */
/* Units are m^3/(kg*s^2)  */
// const double G = 0.1;  /* Gravitational constant. */
/* Units are m^3/(kg*s^2)  */

typedef double vect_t[DIM];  /* Vector type for position, etc. */

struct particle_s {
	double m;  /* Mass     */
	vect_t s;  /* Position */
	vect_t v;  /* Velocity */
};

// 添加全局变量用于控制线程
int current_step;
int total_steps;
pthread_mutex_t step_mutex = PTHREAD_MUTEX_INITIALIZER;
pthread_cond_t step_cond = PTHREAD_COND_INITIALIZER;
int threads_done = 0;

// 修改线程数据结构
struct thread_data {
    int thread_id;
    int n_threads;
    int n_particles;
    struct particle_s *particles;
    vect_t *forces;
    double delta_t;
    pthread_barrier_t *barrier;
    int *current_step_p;    // 添加当前步数指针
    int output_freq;        // 添加输出频率
};

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

// 优化的线程工作函数
void* Thread_work(void* arg) {
    struct thread_data* tdata = (struct thread_data*)arg;
    int my_first = (tdata->thread_id * tdata->n_particles) / tdata->n_threads;
    int my_last = ((tdata->thread_id + 1) * tdata->n_particles) / tdata->n_threads;
    
    while (*tdata->current_step_p <= total_steps) {
        // 计算力
        for (int i = my_first; i < my_last; i++) {
            Compute_force(i, tdata->forces, tdata->particles, tdata->n_particles);
        }
        
        pthread_barrier_wait(tdata->barrier);
        
        // 更新粒子位置
        for (int i = my_first; i < my_last; i++) {
            Update_part(i, tdata->forces, tdata->particles, tdata->n_particles, tdata->delta_t);
        }
        
        pthread_barrier_wait(tdata->barrier);
        
        // 主线程处理输出
        if (tdata->thread_id == 0) {
            if (*tdata->current_step_p % tdata->output_freq == 0) {
                // 处理输出
                printf("Step %d\n", *tdata->current_step_p);
            }
            (*tdata->current_step_p)++;
        }
        
        pthread_barrier_wait(tdata->barrier);
    }
    
    return NULL;
}

/*--------------------------------------------------------------------*/
int main(int argc, char *argv[]) {
    int n_threads;           // 添加线程数变量声明
    int n;                   /* Number of particles */
    int n_steps;
    double delta_t;
    int output_freq;
    char g_i;
    double start, finish;    /* 计时变量 */
    struct particle_s *curr;
    vect_t *forces;
    
    // 获取参数
    Get_args(argc, argv, &n_threads, &n, &n_steps, &delta_t, &output_freq, &g_i);
    
    // 修正内存分配的类型转换
    curr = (struct particle_s *)malloc(n * sizeof(struct particle_s));
    forces = (vect_t *)malloc(n * sizeof(vect_t));
    
    if (curr == NULL || forces == NULL) {
        fprintf(stderr, "内存分配失败\n");
        exit(1);
    }

    // 初始化barrier（现在n_threads已定义）
    pthread_barrier_init(&barrier, NULL, n_threads);

    if (g_i == 'i')
        Get_init_cond(curr, n);
    else
        Gen_init_cond(curr, n);

    // 初始化
    current_step = 1;
    total_steps = n_steps;

    // 创建线程
    pthread_t* thread_handles = malloc(n_threads * sizeof(pthread_t));
    struct thread_data* thread_data_array = malloc(n_threads * sizeof(struct thread_data));
    
    GET_TIME(start);
    
    // 只创建一次线程
    for (int i = 0; i < n_threads; i++) {
        thread_data_array[i].thread_id = i;
        thread_data_array[i].n_threads = n_threads;
        thread_data_array[i].n_particles = n;
        thread_data_array[i].particles = curr;
        thread_data_array[i].forces = forces;
        thread_data_array[i].delta_t = delta_t;
        thread_data_array[i].barrier = &barrier;
        thread_data_array[i].current_step_p = &current_step;
        thread_data_array[i].output_freq = output_freq;
        
        pthread_create(&thread_handles[i], NULL, Thread_work, 
                     (void*)&thread_data_array[i]);
    }
    
    // 等待所有线程完成
    for (int i = 0; i < n_threads; i++) {
        pthread_join(thread_handles[i], NULL);
    }
    
    GET_TIME(finish);
    
    // 清理资源
    pthread_barrier_destroy(&barrier);
    pthread_mutex_destroy(&step_mutex);
    pthread_cond_destroy(&step_cond);
    free(thread_handles);
    free(thread_data_array);

    printf("Elapsed time = %e seconds\n", finish - start);

    free(curr);
    free(forces);
    return 0;
}  /* main */


/*---------------------------------------------------------------------
 * Function: Usage
 * Purpose:  Print instructions for command-line and exit
 * In arg:
 *    prog_name:  the name of the program as typed on the command-line
 */
void Usage(char *prog_name) {
	fprintf(stderr, "usage: %s <number of particles> <number of timesteps>\n",
	        prog_name);
	fprintf(stderr, "   <size of timestep> <output frequency>\n");
	fprintf(stderr, "   <g|i>\n");
	fprintf(stderr, "   'g': program should generate init conds\n");
	fprintf(stderr, "   'i': program should get init conds from stdin\n");

	exit(0);
}  /* Usage */


/*---------------------------------------------------------------------
 * Function:  Get_args
 * Purpose:   Get command line args
 * In args:
 *    argc:            number of command line args
 *    argv:            command line args
 * Out args:
 *    n_p:             pointer to n, the number of particles
 *    n_steps_p:       pointer to n_steps, the number of timesteps
 *    delta_t_p:       pointer to delta_t, the size of each timestep
 *    output_freq_p:   pointer to output_freq, which is the number of
 *                     timesteps between steps whose output is printed
 *    g_i_p:           pointer to char which is 'g' if the init conds
 *                     should be generated by the program and 'i' if
 *                     they should be read from stdin
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

#  ifdef DEBUG
	printf("n = %d\n", *n_p);
	printf("n_steps = %d\n", *n_steps_p);
	printf("delta_t = %e\n", *delta_t_p);
	printf("output_freq = %d\n", *output_freq_p);
	printf("g_i = %c\n", *g_i_p);
#  endif
}  /* Get_args */


/*---------------------------------------------------------------------
 * Function:  Get_init_cond
 * Purpose:   Read in initial conditions:  mass, position and velocity
 *            for each particle
 * In args:
 *    n:      number of particles
 * Out args:
 *    curr:   array of n structs, each struct stores the mass (scalar),
 *            position (vector), and velocity (vector) of a particle
 */
void Get_init_cond(struct particle_s curr[], int n) {
	int part;

	printf("For each particle, enter (in order):\n");
	printf("   its mass, its x-coord, its y-coord, ");
	printf("its x-velocity, its y-velocity\n");
	for (part = 0; part < n; part++) {
		scanf("%lf", &curr[part].m);
		scanf("%lf", &curr[part].s[X]);
		scanf("%lf", &curr[part].s[Y]);
		scanf("%lf", &curr[part].v[X]);
		scanf("%lf", &curr[part].v[Y]);
	}
}  /* Get_init_cond */

/*---------------------------------------------------------------------
 * Function:  Gen_init_cond
 * Purpose:   Generate initial conditions:  mass, position and velocity
 *            for each particle
 * In args:
 *    n:      number of particles
 * Out args:
 *    curr:   array of n structs, each struct stores the mass (scalar),
 *            position (vector), and velocity (vector) of a particle
 *
 * Note:      The initial conditions place all particles at
 *            equal intervals on the nonnegative x-axis with
 *            identical masses, and identical initial speeds
 *            parallel to the y-axis.  However, some of the
 *            velocities are in the positive y-direction and
 *            some are negative.
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
//    if (random()/((double) RAND_MAX) >= 0.5)
		if (part % 2 == 0)
			curr[part].v[Y] = speed;
		else
			curr[part].v[Y] = -speed;
	}
}  /* Gen_init_cond */


/*---------------------------------------------------------------------
 * Function:  Output_state
 * Purpose:   Print the current state of the system
 * In args:
 *    curr:   array with n elements, curr[i] stores the state (mass,
 *            position and velocity) of the ith particle
 *    n:      number of particles
 */
void Output_state(double time, struct particle_s curr[], int n) {
	int part;
	printf("%.2f\n", time);
	for (part = 0; part < n; part++) {
//    printf("%.3f ", curr[part].m);
		printf("%3d %10.3e ", part, curr[part].s[X]);
		printf("  %10.3e ", curr[part].s[Y]);
		printf("  %10.3e ", curr[part].v[X]);
		printf("  %10.3e\n", curr[part].v[Y]);
	}
	printf("\n");
}  /* Output_state */


/*---------------------------------------------------------------------
 * Function:  Compute_force
 * Purpose:   Compute the total force on particle part.  Exploit
 *            the symmetry (force on particle i due to particle k)
 *            = -(force on particle k due to particle i) to also
 *            calculate partial forces on other particles.
 * In args:
 *    part:   the particle on which we're computing the total force
 *    curr:   current state of the system:  curr[i] stores the mass,
 *            position and velocity of the ith particle
 *    n:      number of particles
 * Out arg:
 *    forces: force[i] stores the total force on the ith particle
 *
 * Note: This function uses the force due to gravitation.  So
 * the force on particle i due to particle k is given by
 *
 *    m_i m_k (s_k - s_i)/|s_k - s_i|^2
 *
 * Here, m_j is the mass of particle j and s_k is its position vector
 * (at time t).
 */
void Compute_force(int part, vect_t forces[], struct particle_s curr[],

                   int n) {
	int k;
	double mg;
	vect_t f_part_k;
	double len, len_3, fact;

#  ifdef DEBUG
	printf("Current total force on particle %d = (%.3e, %.3e)\n",
	       part, forces[part][X], forces[part][Y]);
#  endif
	forces[part][X] = forces[part][Y] = 0.0;
	for (k = 0; k < n; k++) {
		if (k != part) {
			/* Compute force on part due to k */
			f_part_k[X] = curr[part].s[X] - curr[k].s[X];
			f_part_k[Y] = curr[part].s[Y] - curr[k].s[Y];
			len = sqrt(f_part_k[X] * f_part_k[X] + f_part_k[Y] * f_part_k[Y]);
			len_3 = len * len * len;
			mg = -G * curr[part].m * curr[k].m;
			fact = mg / len_3;
			f_part_k[X] *= fact;
			f_part_k[Y] *= fact;
#     ifdef DEBUG
			printf("Force on particle %d due to particle %d = (%.3e, %.3e)\n",
			       part, k, f_part_k[X], f_part_k[Y]);
#     endif

			/* Add force in to total forces */
			forces[part][X] += f_part_k[X];
			forces[part][Y] += f_part_k[Y];
		}
	}
}  /* Compute_force */


/*---------------------------------------------------------------------
 * Function:  Update_part
 * Purpose:   Update the velocity and position for particle part
 * In args:
 *    part:    the particle we're updating
 *    forces:  forces[i] stores the total force on the ith particle
 *    n:       number of particles
 *
 * In/out arg:
 *    curr:    curr[i] stores the mass, position and velocity of the
 *             ith particle
 *
 * Note:  This version uses Euler's method to update both the velocity
 *    and the position.
 */
void Update_part(int part, vect_t forces[], struct particle_s curr[],

                 int n, double delta_t) {
	double fact = delta_t / curr[part].m;

#  ifdef DEBUG
	printf("Before update of %d:\n", part);
	printf("   Position  = (%.3e, %.3e)\n", curr[part].s[X], curr[part].s[Y]);
	printf("   Velocity  = (%.3e, %.3e)\n", curr[part].v[X], curr[part].v[Y]);
	printf("   Net force = (%.3e, %.3e)\n", forces[part][X], forces[part][Y]);
#  endif
	curr[part].s[X] += delta_t *curr[part].v[X];
	curr[part].s[Y] += delta_t *curr[part].v[Y];
	curr[part].v[X] += fact * forces[part][X];
	curr[part].v[Y] += fact * forces[part][Y];
#  ifdef DEBUG
	printf("Position of %d = (%.3e, %.3e), Velocity = (%.3e,%.3e)\n",
	       part, curr[part].s[X], curr[part].s[Y],
	       curr[part].v[X], curr[part].v[Y]);
#  endif
// curr[part].s[X] += delta_t * curr[part].v[X];
// curr[part].s[Y] += delta_t * curr[part].v[Y];
}  /* Update_part */


/*---------------------------------------------------------------------
 * Function:  Compute_energy
 * Purpose:   Compute the kinetic and potential energy in the system
 * In args:
 *    curr:   current state of the system, curr[i] stores the mass,
 *            position and velocity of the ith particle
 *    n:      number of particles
 * Out args:
 *    kin_en_p: pointer to kinetic energy of system
 *    pot_en_p: pointer to potential energy of system
 */
void Compute_energy(struct particle_s curr[], int n, double *kin_en_p,
                    double *pot_en_p) {
	int i, j;
	vect_t diff;
	double pe = 0.0, ke = 0.0;
	double dist, speed_sqr;

	for (i = 0; i < n; i++) {
		speed_sqr = curr[i].v[X] * curr[i].v[X] + curr[i].v[Y] * curr[i].v[Y];
		ke += curr[i].m * speed_sqr;
	}
	ke *= 0.5;

	for (i = 0; i < n - 1; i++) {
		for (j = i + 1; j < n; j++) {
			diff[X] = curr[i].s[X] - curr[j].s[X];
			diff[Y] = curr[i].s[Y] - curr[j].s[Y];
			dist = sqrt(diff[X] * diff[X] + diff[Y] * diff[Y]);
			pe += -G * curr[i].m * curr[j].m / dist;
		}
	}

	*kin_en_p = ke;
	*pot_en_p = pe;
}  /* Compute_energy */
