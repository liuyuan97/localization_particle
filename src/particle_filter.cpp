/*
 * particle_filter.cpp
 *
 *  Created on: Dec 12, 2016
 *      Author: Tiffany Huang
 */

#include <random>
#include <algorithm>
#include <iostream>
#include <numeric>
#include <math.h> 
#include <iostream>
#include <sstream>
#include <string>
#include <iterator>

#include "particle_filter.h"

using namespace std;

void ParticleFilter::init(double x, double y, double theta, double std[]) {
	
    default_random_engine gen;
    // Set the number of particles
    num_particles = 100;
	// creates a normal distribution for x, y and theta.
	normal_distribution<double> dist_x(x, std[0]);
	normal_distribution<double> dist_y(y, std[1]);
	normal_distribution<double> dist_theta(theta, std[2]);

	// initialize the particles
	for (int indx = 0; indx < num_particles; indx ++) {
		struct Particle new_particle;
		new_particle.id = indx;
		new_particle.x = dist_x(gen);
		new_particle.y = dist_y(gen);
		new_particle.theta = dist_theta(gen);
		new_particle.weight = 1.0;
		particles.push_back (new_particle);
	}
	is_initialized = true;
	return;
}

void ParticleFilter::prediction(double delta_t, double std_pos[], double velocity, double yaw_rate) {
	
	// creates a normal distribution for x, y and theta.
	default_random_engine gen;
	normal_distribution<double> dist_x(0, std_pos[0]);
	normal_distribution<double> dist_y(0, std_pos[1]);
	normal_distribution<double> dist_theta(0, std_pos[2]);

	if (fabs(yaw_rate) <= 0.001) {
		for (int indx = 0; indx < num_particles; indx ++) {
			particles[indx].x += velocity * cos(particles[indx].theta) * delta_t + dist_x(gen);
			particles[indx].y += velocity * sin(particles[indx].theta) * delta_t + dist_y(gen);
			particles[indx].theta += dist_theta(gen);
		}

	} else {
		for (int indx = 0; indx < num_particles; indx ++ ) {
			double theta = particles[indx].theta;
			particles[indx].x += velocity /yaw_rate  * (sin(theta + yaw_rate * delta_t) - sin(theta)) + dist_x(gen);
			particles[indx].y += velocity /yaw_rate  * (-cos(theta + yaw_rate * delta_t) + cos(theta))+ dist_y(gen);
			particles[indx].theta += yaw_rate * delta_t + dist_theta(gen);
		}
	}
	return;
}

void ParticleFilter::dataAssociation(std::vector<LandmarkObs> predicted, std::vector<LandmarkObs>& observations) {
	
	// for each obervation, find the corresponding landmark
	for (uint indx = 0; indx < observations.size(); indx ++) {
		double min_distance = std::numeric_limits<double>::max();
		int min_id = 0;
		for (uint indx1 = 0; indx1 < predicted.size(); indx1++) {
			double distance = dist(observations[indx].x, observations[indx].y, predicted[indx1].x, predicted[indx1].y);
			if (min_distance > distance) {
				min_distance = distance;
				min_id = indx1;
			}
		}
		observations[indx].id = min_id;
	}
	return;
}

void ParticleFilter::updateWeights(double sensor_range, double std_landmark[], 
		const std::vector<LandmarkObs> &observations, const Map &map_landmarks) {

	int obsv_num = observations.size(); // get the observation size
	std::vector<LandmarkObs> obsv_map;  // obervations in the map coordinate

	double inv_num = 1.0 / (2.0 * M_PI * std_landmark[0] * std_landmark[1]);
	double x_var = std_landmark[0] * std_landmark[0];
	double y_var = std_landmark[1] * std_landmark[1];

	for (int indx = 0; indx < num_particles; indx ++ ) {
		double xp    = particles[indx].x;
		double yp    = particles[indx].y;
		double theta = particles[indx].theta;

		// transform the obervations from vehicle coordinate to the mape coordinate
		for (int indx1 = 0; indx1 < obsv_num; indx1 ++) {
			struct LandmarkObs new_obsv;
			new_obsv.x = xp + cos(theta) * observations[indx1].x - sin(theta) * observations[indx1].y;
			new_obsv.y = yp + sin(theta) * observations[indx1].x + cos(theta) * observations[indx1].y;
			obsv_map.push_back (new_obsv);
		}

		std::vector<LandmarkObs> predicted_lm;
		predicted_lm.clear();

		// find all landmakr in the range of the particle
		for (uint indx1 = 0; indx1 < map_landmarks.landmark_list.size(); indx1 ++) {
			double lm_dist = dist(xp, yp, map_landmarks.landmark_list[indx1].x_f, map_landmarks.landmark_list[indx1].y_f);
			if (lm_dist < sensor_range) {
				LandmarkObs new_lm; 
				new_lm.x = map_landmarks.landmark_list[indx1].x_f;
				new_lm.y = map_landmarks.landmark_list[indx1].y_f;
				new_lm.id = map_landmarks.landmark_list[indx1].id_i;
				predicted_lm.push_back(new_lm);
			}
		}
		// do the observation and map data association
		dataAssociation(predicted_lm, obsv_map);

		// compute the particle weight 
		particles[indx].weight = 1.0;
		for (int indx1 = 0; indx1 < obsv_num; indx1 ++) {
			double new_weight;
			double delta_x = obsv_map[indx1].x - predicted_lm[obsv_map[indx1].id].x;
			double delta_y = obsv_map[indx1].y - predicted_lm[obsv_map[indx1].id].y;
			new_weight = exp( -pow( delta_x, 2.0)  / (2.0 * x_var) - pow( delta_y, 2.0)  / (2.0 * y_var));
		
			new_weight *= inv_num;

			particles[indx].weight *= new_weight;
		}
		weights.push_back(particles[indx].weight);
		
		// save the association results for debug purpose
		std::vector<int> associations; 
        std::vector<double> sense_x; 
        std::vector<double> sense_y;

		for (int indx1 = 0; indx1 < obsv_num; indx1 ++) {
			associations.push_back(predicted_lm[obsv_map[indx1].id].id);
			sense_x.push_back(obsv_map[indx1].x);
			sense_y.push_back(obsv_map[indx1].y);
		}
		SetAssociations(particles[indx], associations, sense_x, sense_y);
		
		obsv_map.clear(); // remove all element in the last particle

	}
	return;
}

void ParticleFilter::resample() {

	std::vector<Particle> new_particles;
	double max_weight =  *std::max_element(weights.begin(), weights.end());

	std::default_random_engine generator;
	std::uniform_real_distribution<double> distribution(0.0,1.0);

	int indx = int(distribution(generator) * (num_particles - 1));
	double beta = 0;
	for (int i = 0; i < num_particles; i ++) {
		beta += distribution(generator) * 2.0 * max_weight;
		while (weights[indx] < beta) {
			beta = beta - weights[indx];
			indx ++;
			if (indx >= num_particles) {
				indx = 0;
			}
		}
		new_particles.push_back(particles[indx]);
	}
	particles = new_particles;
	weights.clear(); 
	return;
}

void ParticleFilter::SetAssociations(Particle& particle, const std::vector<int>& associations, 
                                     const std::vector<double>& sense_x, const std::vector<double>& sense_y)
{
    //particle: the particle to assign each listed association, and association's (x,y) world coordinates mapping to
    // associations: The landmark id that goes along with each listed association
    // sense_x: the associations x mapping already converted to world coordinates
    // sense_y: the associations y mapping already converted to world coordinates

    //Clear the previous associations
	particle.associations.clear();
	particle.sense_x.clear();
	particle.sense_y.clear();

    particle.associations= associations;
    particle.sense_x = sense_x;
    particle.sense_y = sense_y;
    return;
}

string ParticleFilter::getAssociations(Particle best)
{
	vector<int> v = best.associations;
	stringstream ss;
    copy( v.begin(), v.end(), ostream_iterator<int>(ss, " "));
    string s = ss.str();
    s = s.substr(0, s.length()-1);  // get rid of the trailing space
    return s;
}
string ParticleFilter::getSenseX(Particle best)
{
	vector<double> v = best.sense_x;
	stringstream ss;
    copy( v.begin(), v.end(), ostream_iterator<float>(ss, " "));
    string s = ss.str();
    s = s.substr(0, s.length()-1);  // get rid of the trailing space
    return s;
}
string ParticleFilter::getSenseY(Particle best)
{
	vector<double> v = best.sense_y;
	stringstream ss;
    copy( v.begin(), v.end(), ostream_iterator<float>(ss, " "));
    string s = ss.str();
    s = s.substr(0, s.length()-1);  // get rid of the trailing space
    return s;
}
