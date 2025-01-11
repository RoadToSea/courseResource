#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <sys/time.h>
#include <omp.h>

/* ��ȡ��ǰʱ��ĺ궨�� */
#define GET_TIME(now) { \
		struct timeval t; \
		gettimeofday(&t, NULL); \
		now = t.tv_sec + t.tv_usec / 1000000.0; \
	}

#define DIM 2  /* ��άϵͳ */
#define X 0    /* x������±� */
#define Y 1    /* y������±� */

const double G = 6.673e-11;  /* �������� */
/* ��λ�� m^3/(kg*s^2) */

/* �������ӽṹ�� */
typedef double vect_t[DIM];  /* �������ͣ���ʾλ�á��ٶȵ� */

struct particle_s {
	double m;  /* ���� */
	vect_t s;  /* λ�� */
	vect_t v;  /* �ٶ� */
};

/* �������� */
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
	int n;                      /* �������� */
	int n_steps;                /* ʱ�䲽�� */
	int step;                   /* ��ǰ���� */
	int part;                   /* ��ǰ���� */
	int output_freq;            /* ���Ƶ�� */
	double delta_t;             /* ʱ�䲽�� */
	double t;                   /* ��ǰʱ�� */
	struct particle_s *curr;    /* ��ǰϵͳ״̬ */
	vect_t *forces;             /* ÿ�������ϵ��� */
	char g_i;                   /* 'g'��ʾ���ɳ�ʼ������'i'��ʾ�ӱ�׼�����ȡ */

#  ifdef COMPUTE_ENERGY
	double kinetic_energy, potential_energy;
#  endif
	double start, finish;       /* ���ڼ�ʱ */

	/* ��ȡ�����в��� */
	Get_args(argc, argv, &n, &n_steps, &delta_t, &output_freq, &g_i);
	curr = malloc(n * sizeof(struct particle_s));
	forces = malloc(n * sizeof(vect_t));
	if (g_i == 'i')
		Get_init_cond(curr, n);
	else
		Gen_init_cond(curr, n);

	GET_TIME(start);

#  ifdef COMPUTE_ENERGY
	/* ���㲢�����ʼ���� */
	Compute_energy(curr, n, &kinetic_energy, &potential_energy);
	printf("��ʼ״̬: PE = %e, KE = %e, ������ = %e\n",
	       potential_energy, kinetic_energy, kinetic_energy + potential_energy);
#  endif

#  ifndef NO_OUTPUT
	/* �����ʼ״̬ */
	Output_state(0, curr, n);
#  endif

	/* ��ʱ��ѭ�� */
	for (step = 1; step <= n_steps; step++) {
		t = step * delta_t;

		/* ���м���ÿ�������ܵ����� */
		#pragma omp parallel for
		for (part = 0; part < n; part++) {
			Compute_force(part, forces, curr, n);
		}

		/* ���и���ÿ�����ӵ�״̬ */
		#pragma omp parallel for
		for (part = 0; part < n; part++) {
			Update_part(part, forces, curr, n, delta_t);
		}

#     ifdef COMPUTE_ENERGY
		/* ���㲢�����ǰ���� */
		Compute_energy(curr, n, &kinetic_energy, &potential_energy);
		printf("�� %d ��: PE = %e, KE = %e, ������ = %e\n",
		       step, potential_energy, kinetic_energy, kinetic_energy + potential_energy);
#     endif

#     ifndef NO_OUTPUT
		/* ÿ��ָ���������һ��״̬ */
		if (step % output_freq == 0)
			Output_state(t, curr, n);
#     endif
	}

	GET_TIME(finish);
	printf("�ܺ�ʱ = %e ��\n", finish - start);

	free(curr);
	free(forces);
	return 0;
}

/*---------------------------------------------------------------------
 * �÷���ʾ����
 * ��ӡ�����в���ʹ��˵�����˳�
 */
void Usage(char *prog_name) {
	fprintf(stderr, "�÷�: %s <��������> <ʱ�䲽��>\n", prog_name);
	fprintf(stderr, "   <ʱ�䲽��> <���Ƶ��>\n");
	fprintf(stderr, "   <g|i>\n");
	fprintf(stderr, "   'g': �����Զ����ɳ�ʼ����\n");
	fprintf(stderr, "   'i': �ӱ�׼�����ȡ��ʼ����\n");

	exit(0);
}

/*---------------------------------------------------------------------
 * ��ȡ�����в���
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
 * �ӱ�׼�����ȡ���ӵĳ�ʼ������������λ�á��ٶȣ�
 */
void Get_init_cond(struct particle_s curr[], int n) {
	int part;

	printf("������ÿ�����ӵ�������λ�ú��ٶȣ�\n");
	for (part = 0; part < n; part++) {
		scanf("%lf", &curr[part].m);
		scanf("%lf", &curr[part].s[X]);
		scanf("%lf", &curr[part].s[Y]);
		scanf("%lf", &curr[part].v[X]);
		scanf("%lf", &curr[part].v[Y]);
	}
}

/*---------------------------------------------------------------------
 * �������ӵĳ�ʼ���������ȷֲ����ٶ������
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
 * ���ϵͳ��ǰ״̬
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
 * ���������ܵ�������
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
 * �������ӵ�״̬��λ�ú��ٶȣ�
 */
void Update_part(int part, vect_t forces[], struct particle_s curr[],

                 int n, double delta_t) {
	curr[part].v[X] += forces[part][X] * delta_t / curr[part].m;
	curr[part].v[Y] += forces[part][Y] * delta_t / curr[part].m;
	curr[part].s[X] += curr[part].v[X] * delta_t;
	curr[part].s[Y] += curr[part].v[Y] * delta_t;
}

/*---------------------------------------------------------------------
 * ����ϵͳ�Ķ��ܺ�����
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
