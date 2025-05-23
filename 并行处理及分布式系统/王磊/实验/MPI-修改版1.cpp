#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <mpi.h>
#include <sys/time.h>

#define DIM 2  /* Two-dimensional system */
#define X 0    /* x-coordinate subscript */
#define Y 1    /* y-coordinate subscript */

const double G = 6.673e-11;  /* Gravitational constant. */
/* Units are m^3/(kg*s^2)  */

typedef double vect_t[DIM];  /* Vector type for position, etc. */

struct particle_s {
	double m;  /* Mass     */
	vect_t s;  /* Position */
	vect_t v;  /* Velocity */
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

/* The argument now should be a double (not a pointer to a double) */
#define GET_TIME(now) { \
		struct timeval t; \
		gettimeofday(&t, NULL); \
		now = t.tv_sec + t.tv_usec/1000000.0; \
	}

int main(int argc, char *argv[]) {
	int n;                      /* Number of particles        */
	int n_steps;                /* Number of timesteps        */
	int step;                   /* Current step               */
	int part;                   /* Current particle           */
	int output_freq;            /* Frequency of output        */
	double delta_t;             /* Size of timestep           */
	double t;                   /* Current Time               */
	struct particle_s *curr;    /* Current state of system    */
	vect_t *forces;             /* Forces on each particle    */
	char g_i;                   /*_G_en or _i_nput init conds */
	double start, finish;       /* For timings                */

	int rank, size;
	MPI_Init(&argc, &argv);
	MPI_Comm_rank(MPI_COMM_WORLD, &rank);
	MPI_Comm_size(MPI_COMM_WORLD, &size);

	if (rank == 0) {
		Get_args(argc, argv, &n, &n_steps, &delta_t, &output_freq, &g_i);
	}

	MPI_Bcast(&n, 1, MPI_INT, 0, MPI_COMM_WORLD);
	MPI_Bcast(&n_steps, 1, MPI_INT, 0, MPI_COMM_WORLD);
	MPI_Bcast(&delta_t, 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);
	MPI_Bcast(&output_freq, 1, MPI_INT, 0, MPI_COMM_WORLD);
	MPI_Bcast(&g_i, 1, MPI_CHAR, 0, MPI_COMM_WORLD);

	int local_n = n / size;
	if (rank < n % size)
		local_n++;

	curr = malloc(local_n * sizeof(struct particle_s));
	forces = malloc(local_n * sizeof(vect_t));

	if (rank == 0) {
		struct particle_s *all_particles = malloc(n * sizeof(struct particle_s));
		if (g_i == 'i')
			Get_init_cond(all_particles, n);
		else
			Gen_init_cond(all_particles, n);

		MPI_Datatype particle_type;
		MPI_Type_contiguous(5, MPI_DOUBLE, &particle_type);
		MPI_Type_commit(&particle_type);

		int *sendcounts = malloc(size * sizeof(int));
		int *displs = malloc(size * sizeof(int));
		for (int i = 0; i < size; i++) {
			sendcounts[i] = n / size;
			if (i < n % size)
				sendcounts[i]++;
			displs[i] = (i > 0) ? displs[i - 1] + sendcounts[i - 1] : 0;
		}

		MPI_Scatterv(all_particles, sendcounts, displs, particle_type, curr, local_n, particle_type, 0, MPI_COMM_WORLD);

		free(all_particles);
		free(sendcounts);
		free(displs);
	} else {
		MPI_Datatype particle_type;
		MPI_Type_contiguous(5, MPI_DOUBLE, &particle_type);
		MPI_Type_commit(&particle_type);

		MPI_Scatterv(NULL, NULL, NULL, particle_type, curr, local_n, particle_type, 0, MPI_COMM_WORLD);
	}

	GET_TIME(start);

	for (step = 1; step <= n_steps; step++) {
		t = step * delta_t;
		for (part = 0; part < local_n; part++)
			Compute_force(part, forces, curr, local_n);
		for (part = 0; part < local_n; part++)
			Update_part(part, forces, curr, local_n, delta_t);

		if (step % output_freq == 0) {
			struct particle_s *all_particles = NULL;
			if (rank == 0) {
				all_particles = malloc(n * sizeof(struct particle_s));
			}

			MPI_Datatype particle_type;
			MPI_Type_contiguous(5, MPI_DOUBLE, &particle_type);
			MPI_Type_commit(&particle_type);

			int *recvcounts = malloc(size * sizeof(int));
			int *displs = malloc(size * sizeof(int));
			for (int i = 0; i < size; i++) {
				recvcounts[i] = n / size;
				if (i < n % size)
					recvcounts[i]++;
				displs[i] = (i > 0) ? displs[i - 1] + recvcounts[i - 1] : 0;
			}

			MPI_Gatherv(curr, local_n, particle_type, all_particles, recvcounts, displs, particle_type, 0, MPI_COMM_WORLD);

			if (rank == 0) {
				Output_state(t, all_particles, n);
				free(all_particles);
			}

			free(recvcounts);
			free(displs);
		}
	}

	GET_TIME(finish);
	if (rank == 0)
		printf("Elapsed time = %e seconds\n", finish - start);

	free(curr);
	free(forces);
	MPI_Finalize();
	return 0;
}

void Usage(char *prog_name) {
	fprintf(stderr, "usage: %s <number of particles> <number of timesteps>\n",
	        prog_name);
	fprintf(stderr, "   <size of timestep> <output frequency>\n");
	fprintf(stderr, "   <g|i>\n");
	fprintf(stderr, "   'g': program should generate init conds\n");
	fprintf(stderr, "   'i': program should get init conds from stdin\n");
	exit(0);
}

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
}

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
			mg = -G * curr[part].m * curr[k].m;
			fact = mg / len_3;
			f_part_k[X] *= fact;
			f_part_k[Y] *= fact;
			forces[part][X] += f_part_k[X];
			forces[part][Y] += f_part_k[Y];
		}
	}
}

void Update_part(int part, vect_t forces[], struct particle_s curr[],

                 int n, double delta_t) {
	double fact = delta_t / curr[part].m;
	curr[part].s[X] += delta_t *curr[part].v[X];
	curr[part].s[Y] += delta_t *curr[part].v[Y];
	curr[part].v[X] += fact * forces[part][X];
	curr[part].v[Y] += fact * forces[part][Y];
}

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
}
