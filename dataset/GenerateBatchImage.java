import org.openscience.cdk.depict.Abbreviations;
import org.openscience.cdk.depict.DepictionGenerator;
import org.openscience.cdk.interfaces.IAtomContainer;
import org.openscience.cdk.interfaces.IChemObjectBuilder;
import org.openscience.cdk.interfaces.IPseudoAtom;
import org.openscience.cdk.silent.SilentChemObjectBuilder;
import org.openscience.cdk.smiles.SmilesParser;

import java.awt.*;
import java.io.BufferedReader;
import java.io.File;
import java.io.FileReader;
import java.math.BigDecimal;
import java.util.ArrayList;
import java.util.List;
import java.util.Random;

//import sun.font.TrueTypeFont;

public class GenerateBatchImage {
    public static void main(String[] args) throws Exception {
        IChemObjectBuilder bldr   = SilentChemObjectBuilder.getInstance();
        SmilesParser smipar = new SmilesParser(bldr);
        String smilesFileName = "./smiles.txt";
        String pngFileName ="./images";
        List<String> errornum = new ArrayList<String>();
        File file = new File(smilesFileName);
        BufferedReader reader = null;
        reader=new BufferedReader(new FileReader(file));
        String temp = reader.readLine();
        Random random = new Random(7);
        int index = 0;
        int num=1;
        String a[]=new String[28];
        {
            a[0]="[3H]";a[1]="[2H]";a[2]="C(=O)O[R4]";a[3]="N[R2]";a[4]="N[R5]";a[5]="=N[R5]";a[6]="P[R3]";a[7]="[X][R3]";a[8]="[R1][R1][R1]";a[9]="[R2][R2][R2]";
            a[10]="[Y][Y]";a[11]="[Y][Y][Y]";a[12]="CCOC(=O)";a[13]="[(CH2)0-2]";a[14]="[(CH2)b]";a[15]="[(CH2)c]";a[16]="[(CH2)d]";a[17]="[(CH2)e]";a[18]="[(CH2)f]";
            a[19]="[(CH2)b]";a[20]="[(CH2)r]";a[21]="[(CH2)p]";a[22]="[(CH2)n]";a[23]="[(C)t]";a[24]="[(C)s]";a[25]="[(C)m]";a[26]="[R]";a[27]="[(CH2)q]";
        }
        String b[]=new String[28];
        {
            b[0]="[T]";b[1]="[D]";b[2]="[CO2R4]";b[3]="[NR2]";b[4]="[R5NH]";b[5]="=[NR5]";b[6]="[PR3]";b[7]="[R3X]";b[8]="(R1)3";b[9]="(R2)3";
            b[10]="[Y2]";b[11]="[Y3]";b[12]="[CO2Et]";b[13]="[(  )0-2]";b[14]="[(  )b]";b[15]="[(  )c]";b[16]="[(  )d]";b[17]="[(  )e]";b[18]="[(  )f]";
            b[19]="[(  )b]";b[20]="[(  )r]";b[21]="[(  )p]";b[22]="[(  )n]";b[23]="[(  )t]";b[24]="[(  )s]";b[25]="[(  )m]";b[26]="[*]";b[27]="[(  )q]";
        }
        while(temp!=null){
            // 做相应的操作
            Abbreviations abrv = new Abbreviations();
            int tag=1;
            num=num+1;
//            if(num>=3){
//                break;
//            }
            String [] terms = temp.replace("\n", "").split("\t");
            String smiles=terms[1];
            if (smiles.startsWith("CC1=N[")){
                smiles=smiles.replaceFirst("C","*");
                System.out.println(smiles);
                tag=1;
            }//特定情况下显示*号
            if (smiles.indexOf("[F,Cl,Br,I]")!=-1){
                smiles=smiles.replace("F,Cl,Br,I","X");
            }//卤素元素替换为X
            if (smiles.indexOf("[*]")!=-1){
                String[] str = { "[A]", "[W]"};
                int random1 = (int) ( Math.random () * 2 );
                System.out.println (str[random1]);
                smiles=smiles.replace("[*]",str[random1]);
            }//*随机为A/W
            if (smiles.indexOf("O[R1]")!=-1 ||smiles.indexOf("O[R2]")!=-1||smiles.indexOf("O[R5]")!=-1||smiles.indexOf("O[R6]")!=-1||smiles.indexOf("O[P][G2]")!=-1||smiles.indexOf("O[R7]")!=-1){
                smiles=smiles.replace("O[R1]","[OR1]").replace("O[R2]","[OR2]").replace("O[R5]","[OR5]").replace("O[R6]","[OR6]").replace("O[R7]","[OR7]").replace("O[P][G2]","[OPG2]");
            }
            for(int i=0; i<a.length;i++) {
                smiles=smiles.replace(a[i],b[i]);
            }//批量替换 临近取代基合并

            abrv.add("*[N+](=O)[O-] O2N");
            abrv.add("*C(=O)([O-]) O2C");
            abrv.add("*C#N CN");
            abrv.add("*C(F)(F)F F3C");
            abrv.add("*=C(F)F CF2");
            if (num % 5 == 0) {
                abrv.add("*C(=O)O HOOC");
                abrv.add("*OC OMe");
            } else {
                abrv.add("*C(=O)O CO2H");
            }

            if (smiles.startsWith("CC1=NN2C") || smiles.startsWith("CC1=[")) {
                smiles = smiles.replaceFirst("C", "*");
                System.out.println(smiles);
                tag = 0;
            }
            if (smiles.startsWith("CC(C)(C)[(") || smiles.startsWith("CC(C)(C)C1=N[") || smiles.startsWith("CC(C)(C)CC1")) {

                smiles = smiles.replace("CC(C)(C)", "*");
                System.out.println(smiles);
                tag = 0;
            }
            //    CC(C)可能为波浪线/CH3/原始情况/箭头
            if (smiles.startsWith("CC(C)C(C") || smiles.startsWith("CC(C)N1") || smiles.startsWith("CC(C)C1") || smiles.startsWith("CC(C)C(=")) {

                smiles = smiles.replace("CC(C)", "*");
                System.out.println(smiles);
                tag = 0;
            }
            //此为特点smiles生成波浪线
            try{
                IAtomContainer mol = smipar.parseSmiles(smiles);
                // maybe we don't want 'Me' in the depiction
//                System.out.println(tag);
                abrv.setEnabled("Me", false);
                abrv.setContractOnHetero(false);
                int numAdded = abrv.apply(mol);
                if(tag==0) {
                    ((IPseudoAtom) mol.getAtom(0)).setAttachPointNum(1); // set _AP1
                }
//                System.out.println(numAdded);
                DepictionGenerator dptgen = new DepictionGenerator();
                // size in px (raster) or mm (vector)
                // annotations are red by default
                float f = (random.nextFloat() +1);
                BigDecimal c  =  new  BigDecimal(f);
                float  f1  =  c.setScale(1,  BigDecimal.ROUND_HALF_UP).floatValue();
                // .withAromaticDisplay()当在键上设置芳香性时，在图表中显示它。.withZoom 指定所需的缩放系数

                DepictionGenerator dpt =dptgen.withZoom(f1)
                        .withBackgroundColor(Color.WHITE)
                        .withAromaticDisplay();
                dpt.depict(mol).writeTo(pngFileName+terms[0]+".png");
                if(index%10000==0){
                    float ratio = index/600000.0f;
                    int progess = (int) ratio * 100;
                    StringBuilder sb = new StringBuilder();
                    for (int i =0 ; i < progess; i++)
                        sb.append("=");
                    System.out.printf("当前进度" + sb.toString()+">"+String.valueOf(ratio)+"\n");
                }
            }catch (Exception e){
                e.printStackTrace();
                System.out.println("出错"+terms[0]);
                errornum.add(terms[0]);
                System.out.printf("当前错误smiles num为：" + errornum);
                continue;
            }finally {
                index++;
                temp = reader.readLine();
            }
        }
    }
}

