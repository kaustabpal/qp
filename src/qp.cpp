#include <iostream>
#include "Eigen/Dense"
#include <vector>

double min(std::vector<double> vec)
{
	double min = 9999999999.99;
	for(int i=0; i<vec.size(); i++)
	{
		if(vec[i]<=min)
		{
			min = vec[i];
		}
	}
	return min;
}

int main()
{
	Eigen::Matrix<double, 2, 2> H { {4, 1}, {1, 2} };
	Eigen::Matrix<double, 2, 1> c {{1},{1}};
	Eigen::Matrix<double, 6, 2> A {{1,1},{1,0},{0,1},{-1, -1},{-1,0},{0,-1}};
	Eigen::Matrix<double, 6, 1> b {{1},{0.7},{0.7},{-1},{0},{0}};

	const int m = A.rows();
	const int n = A.cols();
	Eigen::Matrix<double, n, 1> x;
	x.setOnes();
	Eigen::Matrix<double, m, 1> lambda;
	lambda.setOnes();
	Eigen::Matrix<double, m, 1> s;
	s.setOnes();
	Eigen::Matrix<double, m, 1> e;
	e.setOnes();
//  ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
	Eigen::Matrix<double, n,1> rd0; 
	rd0 = H*x + c + A.transpose()*lambda;
	Eigen::Matrix<double, m,1> rp0; 
	rp0 = s + A*x -b;
	auto mu = (lambda.array()*s.array()).sum()/m;
	Eigen::Matrix<double, m, 1> rc; 
	rc = Eigen::Matrix<double, m, m>(s.asDiagonal()) * Eigen::Matrix<double, m, m>(lambda.asDiagonal()) * e;
	int i =0;
	double eps = 1e-16;
	int maxiter = 200;
	while( i<=maxiter && rd0.norm() >= eps && rp0.norm() >= eps && abs(mu) >= eps )
	{
		Eigen::Matrix<double, n+m+m, n+m+m> big;
		big.setZero();
		big.block<H.rows(), H.cols()>(0,0)=H;
		big.block<A.cols(), A.rows()>(0, H.cols()) = A.transpose();	
		big.block<A.rows(), A.cols()>(H.rows(),0) = A;
		big.block<m,m>(H.rows(), n+m).setIdentity();
		big.block<s.rows(), s.rows()>(n+m,n) = s.asDiagonal();
		big.block<lambda.rows(), lambda.rows()>(n+m,n+m) = lambda.asDiagonal();
//  ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
		Eigen::Matrix<double, rd0.rows()+rp0.rows()+rc.rows(),1> r;
		r << -rd0, -rp0, -rc;
		Eigen::Matrix<double, n+m+m, 1> aff = big.householderQr().solve(r);
		auto xaff = aff(Eigen::seq(0,n-1));
		auto lambdaff = aff(Eigen::seq(n,n+m-1));
		auto saff = aff(Eigen::seq(n+m,n+m+m-1));

		double alphaaff = 1;
		std::vector<int> indexfindz;
		std::vector<double> lambda_temp;
		std::vector<double> lambdaff_temp;
		std::vector<double> div_temp;
		for(int i=0; i<lambdaff.rows(); i++)
		{
			if(lambdaff[i]<0)
			{
				indexfindz.push_back(i);
			}
		}
		if(indexfindz.empty() == 0)
		{
			for(int i=0; i<indexfindz.size(); i++)
			{
				lambda_temp.push_back(lambda[indexfindz[i]]);
			}	
			for(int i=0; i<indexfindz.size(); i++)
			{
				lambdaff_temp.push_back(lambdaff[indexfindz[i]]);
			}
			for(int i=0; i<indexfindz.size(); i++)
			{
				div_temp.push_back(-lambda_temp[i]/lambdaff_temp[i]);	
			}
			alphaaff = min(std::vector<double> {alphaaff,min(div_temp)});
		}
		std::vector<int> indexfinds;
		std::vector<double> s_temp;
		std::vector<double> saff_temp;
		div_temp.clear();
		for(int i=0; i<saff.rows(); i++)
		{
			if(saff[i]<0)
			{
				indexfinds.push_back(i);
			}
		}
		if(indexfinds.empty() == 0)
		{
			for(int i=0; i<indexfinds.size(); i++)
			{
				s_temp.push_back(s[indexfinds[i]]);
			}	
			for(int i=0; i<indexfinds.size(); i++)
			{
				saff_temp.push_back(saff[indexfinds[i]]);
			}
			for(int i=0; i<indexfinds.size(); i++)
			{
				div_temp.push_back(-s_temp[i]/saff_temp[i]);	
			}
			alphaaff = min(std::vector<double> {alphaaff,min(div_temp)});
		}
		double muaff = ((s + (alphaaff*saff.array()).matrix()).transpose()*(lambda + (alphaaff*lambdaff.array()).matrix())/m)[0];
		double cent = std::pow((muaff/mu),3);
		auto imprc = rc + (saff.array()*lambdaff.array()).matrix() - cent*mu*e;
		Eigen::Matrix<double, rd0.rows()+rp0.rows()+rc.rows(),1> deltars;
		deltars << -rd0, -rp0, -imprc;
		Eigen::Matrix<double, n+m+m, 1> deltas = big.householderQr().solve(deltars); 
		auto deltax = deltas(Eigen::seq(0,n-1));
		auto deltalambda = deltas(Eigen::seq(n,n+m-1));
		auto ddeltas= deltas(Eigen::seq(n+m,n+m+m-1));
		double stepsizealpha = 1;

		indexfindz.clear();
		lambda_temp.clear();
		std::vector<double> deltalambda_temp;
		div_temp.clear();
		for(int i=0; i<deltalambda.rows(); i++)
		{
			if(deltalambda[i]<0)
			{
				indexfindz.push_back(i);
			}
		}
		if(indexfindz.empty() == 0)
		{
			for(int i=0; i<indexfindz.size(); i++)
			{
				lambda_temp.push_back(lambda[indexfindz[i]]);
			}	
			for(int i=0; i<indexfindz.size(); i++)
			{
				deltalambda_temp.push_back(deltalambda[indexfindz[i]]);
			}
			for(int i=0; i<indexfindz.size(); i++)
			{
				div_temp.push_back(-lambda_temp[i]/deltalambda_temp[i]);	
			}
			stepsizealpha = min(std::vector<double> {stepsizealpha,min(div_temp)});
		}
		indexfinds.clear();
		s_temp.clear();
		std::vector<double> ddeltas_temp;
		div_temp.clear();
		for(int i=0; i<ddeltas.rows(); i++)
		{
			if(ddeltas[i]<0)
			{
				indexfinds.push_back(i);
			}
		}
		if(indexfinds.empty() == 0)
		{
			for(int i=0; i<indexfinds.size(); i++)
			{
				s_temp.push_back(s[indexfinds[i]]);
			}	
			for(int i=0; i<indexfinds.size(); i++)
			{
				ddeltas_temp.push_back(ddeltas[indexfinds[i]]);
			}
			for(int i=0; i<indexfinds.size(); i++)
			{
				div_temp.push_back(-s_temp[i]/ddeltas_temp[i]);	
			}
			stepsizealpha = min(std::vector<double> {stepsizealpha,min(div_temp)});
		}
		// compute new points
		x = x + stepsizealpha*deltax;
		lambda = lambda + stepsizealpha*deltalambda;
		s = s + stepsizealpha*ddeltas;
		// update residual values and mu
		rd0 = H*x + c + A.transpose()*lambda;
		rp0 = s + A*x -b;
		rc = Eigen::Matrix<double, m, m>(s.asDiagonal()) * Eigen::Matrix<double, m, m>(lambda.asDiagonal()) * e;
		mu = (lambda.array()*s.array()).sum()/m;
		i = i+1;
	}
	auto cost = ((0.5*x.transpose())*(H*x))+(c.transpose()*x);
	std::cout<<"cost ="<<cost<<"\n";
	std::cout<<"x = \n"<<x<<"\n";
	return 0;
}

