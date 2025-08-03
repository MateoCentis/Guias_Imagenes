#include "CImg.h"
#include "PDI_functions.h"
#include <iostream>

using namespace cimg_library;
using namespace std;

const float rojo[] = {255, 0, 0};
const float blanco[] = {1, 1, 1};
const float negro[] = {0, 0, 0};

CImg<float> producto(CImg<float>img, CImg<float>mascara){
	CImg<float> res(img);
	for (int j=0; j<img.height(); j++){
		for (int i=0; i<img.width(); i++){
			res(i,j)= img(i,j) * mascara(i,j);
		}
	}
	return res;
}


float promedio_color(CImg<float>&img){
	float c = 0;
	float acum = 0;
	for (int j=0; j<img.height(); j++){
		for (int i=0; i<img.width(); i++){
			if (img(i,j)>0 && img(i,j)<100){
				acum += img(i,j);
				c++;
			}
		}
	}
	return acum * 1.0 /c;
}

void invertir(CImg<float>&img){
	float max = img.max();
	for (int j=0; j<img.height(); j++){
		for (int i=0; i<img.width(); i++){
			img(i,j)= max-img(i,j);
		}
	}
	return;
}

void obtener_extremos(CImg<float> img, int &max_x, int &max_y, int &min_x, int &min_y, int w, int h){
	for (int j=0; j<h; j++){
		for (int i=0; i<img.width(); i++){
			if (img(i,j)==1){
				min_y = j;
				break;
			}
				
		}
	}
	for (int j=h-1; j>=0; j--){
		for (int i=0; i<img.width(); i++){
			if (img(i,j)==1){
				max_y = j;
				break;
			}
			
		}
	}
	for (int i=0; i<img.width(); i++){
		for (int j=0; j<h; j++){
			if (img(i,j)==1){
				min_x = i;
				break;
			}
		}
	}
	for (int i=w-1; i>=0; i--){
		for (int j=0; j<h; j++){
			if (img(i,j)==1){
				max_x = i;
				break;
			}
		}
	}
	cout<<max_x<<" "<<max_y<<" "<<min_x<<" "<<min_y<<endl;
}

void segmentar(CImg<float>&tono, int i, int j,  int c){
	int w= tono.width();
	int h= tono.height();
	if ((i<0)||(i>=w)||(j<0)||(j>=h))
		return;
	if (tono(i,j)==1)
		tono(i,j)=c;
	if (((j+1)<h)&&(tono(i,j+1)==1))
		segmentar(tono, i, j+1, c);
	if (((i+1)<w)&&(tono(i+1,j)==1))	
		segmentar(tono, i+1, j, c);
	if (((i-1)>=0)&&(tono(i-1,j)==1))	
		segmentar(tono, i-1, j, c);
	if (((j-1)>=0)&&(tono(i,j-1)==1))	
		segmentar(tono, i, j-1, c);
	return;
}

CImg<double> obtener_moscas(CImg<float> img, int w, int h){
	CImg<float> moscas = img.get_threshold(70);
	moscas.erode(15).display();
	invertir(moscas);
	moscas.display();
	return moscas;
}

void contar_moscas(CImg<float> img, int &cantidad, int w, int h){
	int c=10;
	for (int j=0; j<h;j++){
		for (int i=0; i<w;i++){
			if (img(i,j)==1){
				segmentar(img, i, j, c);
				c++;
			}
		}
	}
//	img.display();
	cantidad = c-10;
	return;
}

int main (int argc, char *argv[]) {
		
		const char* filename = cimg_option("-i","./Platos01.jpg","Image file\n");
		CImg<float> img(filename);

		int w = img.width();
		int h = img.height();
	
		img.display();
			
		CImg<float> img_hsi = img.get_RGBtoHSI();
		
		CImgList<float> list_hsi(img_hsi);
		for (int i=0; i<3; ++i){
			CImg<float> c = img_hsi.get_channel(i);
			list_hsi.push_back(c);
		}
		
		//normalizar H, S, e I:
		list_hsi(1) = list_hsi(1)/360.0 * 255;
		list_hsi(2) *= 255;
		list_hsi(3) *= 255;
		
		CImg<float> buscando_plato = list_hsi(3).get_threshold(210);
		buscando_plato.erode(3).dilate(3);
		buscando_plato.display();
		
		int max_x =0, max_y =0, min_x =0, min_y=0;
		
		obtener_extremos(buscando_plato, max_x, max_y, min_x, min_y, w, h);
		
		int c_x = (max_x+min_x)/2, c_y = (max_y+min_y)/2;
		
	//	list_hsi(1).display();
	//	list_hsi(2).display();
//		list_hsi(3).display();
//		
		CImg<float> intensidad(list_hsi(3)); 
		
		CImg<float> moscas = obtener_moscas(intensidad, w, h);
		
		CImg<float> mascara_plato(w,h);
		mascara_plato.draw_circle(c_x, c_y, 440, blanco);
		mascara_plato.draw_circle(c_x, c_y, 310, negro);
		mascara_plato.display();
		CImg<float> solo_plato_gris = producto(moscas, mascara_plato);
		solo_plato_gris.display();
		
		CImg<float> mascara_sopa(w,h); 	
		mascara_sopa.draw_circle(c_x, c_y, 300, blanco);
		//mascara_sopa.display();
		CImg<float> solo_sopa_azul = producto(img.get_channel(2), mascara_sopa);
		//solo_sopa_azul.display();
		
		int zapallo = 1;
		float prom = promedio_color(solo_sopa_azul);
		if (prom>10)
			zapallo =0;
		
		//cout<<"Color promedio: "<<prom;
		
		CImg<float> solo_sopa_gris = producto(moscas, mascara_sopa);
		
		if (zapallo)
			cout<<"La sopa es de zapallo"<<endl;
		else
			cout<<"La sopa es de la casa"<<endl;
		
		int cantidad_total = 0;
		contar_moscas(moscas, cantidad_total, w, h);
		cout<<"Cantidad total de moscas: "<<cantidad_total<<endl;
		
		int cantidad_sopa = 0;
		contar_moscas(solo_sopa_gris, cantidad_sopa, w, h);
		
		int cantidad_plato = 0;
		contar_moscas(solo_plato_gris, cantidad_plato, w, h);
		cout<<"Cantidad de moscas en el plato: "<<cantidad_plato + cantidad_sopa<<endl;
		
		cout<<"Cantidad de moscas en la sopa: "<<cantidad_sopa<<endl;

		if ((zapallo && cantidad_sopa <= 3) || (!zapallo && cantidad_sopa <= 4))
			cout<<"El plato esta bien servido"<<endl;
		else
			cout<<"El plato esta mal servido"<<endl;
		getchar();
		return 0;	
		
}

