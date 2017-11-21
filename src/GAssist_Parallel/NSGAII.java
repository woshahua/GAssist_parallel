package GAssist_Parallel;
import java.util.ArrayList;

public class NSGAII {
  public static final double INF=Double.POSITIVE_INFINITY; //define Infinity for crowding distance
  public int popSize=0;
  public int[] population_rank;
  public int[] population_count;
  public double[] crowding_dist;
  public ArrayList <ArrayList<Integer>> dominate_solutions=new ArrayList< ArrayList <Integer> > ();
  public ArrayList <ArrayList<Integer>> population_fronts=new ArrayList< ArrayList <Integer> > ();
  

  //  constructor for NSGA2 method  
  NSGAII(int pop_num){
    popSize=pop_num;
    population_rank=new int[popSize];
    population_count=new int[popSize];
    crowding_dist=new double[popSize];
    for(int i=0;i<popSize;i++){
      population_count[i]=0;
      population_rank[i]=0;
      crowding_dist[i]=0;
    }
  }
  
  public void rank(double [] f1,double [] f2, double [] f3){ 
    ArrayList <Integer> front_1 = new ArrayList <Integer> ();
    for(int i=0;i<popSize;i++){
      ArrayList <Integer> temp=new ArrayList <Integer> ();
      ArrayList <Double> dominated_solutions1 = new ArrayList <Double> ();
      ArrayList <Double> dominated_solutions2 = new ArrayList <Double> ();
      for(int j=0;j<popSize;j++){
        if(i!=j){
          if((f1[j]<f1[i]&&f2[j]<=f2[i])||(f1[j]<=f1[i]&&f2[j]<f2[i])){
            boolean is_same = false;
            for(int k = 0; k< dominated_solutions1.size(); k++){
              if(f1[j] == dominated_solutions1.get(k) && f2[j] == dominated_solutions2.get(k)){
                // if there is already a same Individual then same
                is_same = true;
              }
            }
            if(is_same == false){
              population_count[i]+=1;
              dominated_solutions1.add(f1[j]);
              dominated_solutions2.add(f2[j]);
            }
          }
          else if((f1[i]<f1[j]&&f2[i]<=f2[j])||(f1[i]<=f1[j]&&f2[i]<f2[j])){
            temp.add(j);
          }
        }
      }
      dominate_solutions.add(temp);
      if(population_count[i]==0){
        front_1.add(i);
      }
      dominated_solutions1.clear();
      dominated_solutions2.clear();
    }
    population_fronts.add(front_1);
    

    //--------------set all population rank------------------------------
    int i=0;
    while (population_fronts.get(i).size()>0){
      ArrayList <Integer> temp=new ArrayList <Integer> ();
      
      //-------make a change to nsga2 to fit the Pittsburgh-LCS
      
      for(int n=0;n<population_fronts.get(i).size()-1;n++){
        for(int m=n+1;m<population_fronts.get(i).size();m++){
          if(f1[population_fronts.get(i).get(n)] == f1[population_fronts.get(i).get(m)] &&
              f2[population_fronts.get(i).get(n)] == f2[population_fronts.get(i).get(m)]){
            if(f3[population_fronts.get(i).get(n)] < f3[population_fronts.get(i).get(m)]){
              temp.add(population_fronts.get(i).get(m));
                  dominate_solutions.get(population_fronts.get(i).get(m)).clear();
                  population_fronts.get(i).remove(m);
                  m--;
            }
            else{
              temp.add(population_fronts.get(i).get(n));
              dominate_solutions.get(population_fronts.get(i).get(n)).clear();
              population_fronts.get(i).remove(n);
              n--;
              break;
            }
          }
        }
      }
     //-------------------------------------------------- 
        
      for(int j:population_fronts.get(i)){
        for(int k: dominate_solutions.get(j)){
          population_count[k]--;
          if(population_count[k]==0){
            temp.add(k);
          }
        }
      }
      for(int j:temp){
        population_rank[j]=i+1;
      }
      i++;
      population_fronts.add(temp);
    }

    
    //-----------------crowding-distance------------------
    i=0;
    while (population_fronts.get(i).size()>0){
      int front_size=population_fronts.get(i).size();
      if(population_fronts.get(i).size() == 2){
        for(int j:population_fronts.get(i)){
          crowding_dist[j]=INF;
        }
      }
      //front only have two member
      else if(population_fronts.get(i).size() == 1){
        crowding_dist[population_fronts.get(i).get(0)]=INF;
      }
      //front only have one member
      else{
        //----------------OBJECT NO.1----------------------
        for(int j=0;j<front_size-1;j++){
          for(int k=j+1;k<front_size;k++){
            int a1=population_fronts.get(i).get(j);
            int a2=population_fronts.get(i).get(k);
            if(f1[a1]>f1[a2]){
              population_fronts.get(i).set(k,a1);
              population_fronts.get(i).set(j,a2);
            }
          }
        }           
       
//        System.out.println(population_fronts.get(i));
        double max_value,min_value;
        max_value=f1[population_fronts.get(i).get(front_size-1)];
        min_value=f1[population_fronts.get(i).get(0)];
//        System.out.println(max_value+" "+min_value);
        
        crowding_dist[population_fronts.get(i).get(front_size-1)]=INF;
        crowding_dist[population_fronts.get(i).get(0)]=INF;
        
        for(int l=1;l<front_size-1;l++){
          if(max_value==min_value){
            crowding_dist[population_fronts.get(i).get(l)]=0;
          }
          else{
          crowding_dist[population_fronts.get(i).get(l)]=(f1[population_fronts.get(i).get(l+1)]
              -f1[population_fronts.get(i).get(l-1)])/(max_value-min_value);
          }
        }
        //-----------------OBJECT NO.2-----------------
        for(int j=0;j<front_size-1;j++){
          for(int k=j+1;k<front_size;k++){
            int a1=population_fronts.get(i).get(j);
            int a2=population_fronts.get(i).get(k);
            if(f2[a1]>f2[a2]){
              population_fronts.get(i).set(k,a1);
              population_fronts.get(i).set(j,a2);
            }
          }
        }
        max_value=f2[population_fronts.get(i).get(front_size-1)];
        min_value=f2[population_fronts.get(i).get(0)];
        
        crowding_dist[population_fronts.get(i).get(front_size-1)]=INF;
        crowding_dist[population_fronts.get(i).get(0)]=INF;
        
        int crowd_count=0;
        for(int l=1;l<front_size-1;l++){
          if(max_value==min_value){
            crowding_dist[population_fronts.get(i).get(l)]+=0;
          }
          else{
          crowding_dist[population_fronts.get(i).get(l)]+=(f2[population_fronts.get(i).get(l+1)]
              -f2[population_fronts.get(i).get(l-1)])/(max_value-min_value);
          }
        }
      }
      i++;
    }
  }
}
